from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


class GatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        ctx.rank = dist.get_rank()
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grads):
        grad_stack = torch.stack([grad.contiguous() for grad in grads], dim=0)
        dist.all_reduce(grad_stack, op=dist.ReduceOp.SUM)
        return grad_stack[ctx.rank]


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor

    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    sizes = [int(size.item()) for size in size_list]
    max_size = max(sizes)

    if tensor.size(0) < max_size:
        padding = torch.zeros(
            max_size - tensor.size(0),
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, padding], dim=0)

    gathered = GatherWithGrad.apply(tensor)
    return torch.cat([gathered[i][:sizes[i]] for i in range(len(sizes))], dim=0)


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not is_distributed():
        return tensor

    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    sizes = [int(size.item()) for size in size_list]
    max_size = max(sizes)

    if tensor.size(0) < max_size:
        padding = torch.zeros(
            max_size - tensor.size(0),
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, padding], dim=0)

    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat([gathered[i][:sizes[i]] for i in range(len(sizes))], dim=0)


def gather_batch_start(local_batch_size: int, device: torch.device) -> int:
    if not is_distributed():
        return 0
    local_size = torch.tensor([local_batch_size], dtype=torch.long, device=device)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    return sum(int(size.item()) for size in size_list[: dist.get_rank()])


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = self.attention(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class PhenotypeMetricModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config: LongTableEncoder1DConfig,
        embedding_matrix: torch.Tensor,
        query_embedding_matrix: torch.Tensor,
        phenotype_scales: torch.Tensor,
        huber_delta: float,
        projection_loss_weight: float,
        transe_loss_weight: float,
        relation_l2_weight: float,
        min_pair_delta: float,
    ):
        super().__init__(config)
        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        hidden_size = config.dim_out if config.dim_out is not None else config.dim
        self.pooling = AttentionPooling(hidden_size)
        self.text_embedding_matrix = embedding_matrix.cpu()
        self.query_embedding_matrix = nn.Parameter(query_embedding_matrix.float(), requires_grad=False)
        self.phenotype_scales = nn.Parameter(phenotype_scales.float(), requires_grad=False)
        self.relation_projection = nn.Sequential(
            nn.Linear(query_embedding_matrix.size(-1), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.huber_delta = float(huber_delta)
        self.projection_loss_weight = float(projection_loss_weight)
        self.transe_loss_weight = float(transe_loss_weight)
        self.relation_l2_weight = float(relation_l2_weight)
        self.min_pair_delta = float(min_pair_delta)
        self.post_init()

    def text_lookup(self, token_ids: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        flat = self.text_embedding_matrix.index_select(0, token_ids.reshape(-1).cpu())
        flat = flat.to(device=device, dtype=dtype, non_blocking=True)
        return flat.view(*token_ids.shape, flat.size(-1))

    def encode_table(
        self,
        item_ids,
        unit_ids,
        value_text_ids,
        times,
        numeric_values,
        numeric_mask,
        seq_mask,
        type_ids,
    ):
        dtype = self.encoder.embedding.item_proj.weight.dtype
        device = self.encoder.embedding.item_proj.weight.device
        hidden_states, hidden_mask = self.encoder(
            item_emb=self.text_lookup(item_ids, dtype, device),
            unit_emb=self.text_lookup(unit_ids, dtype, device),
            value_emb=self.text_lookup(value_text_ids, dtype, device),
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            seq_mask=seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )
        hidden_states = self.adapter(hidden_states, hidden_mask)
        pooled_mask = torch.ones(hidden_states.shape[:2], dtype=hidden_mask.dtype, device=hidden_mask.device)
        return self.pooling(hidden_states, pooled_mask)

    def relation_vectors(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        query_embeddings = self.query_embedding_matrix.to(device=device, dtype=dtype)
        return F.normalize(self.relation_projection(query_embeddings), dim=-1)

    def _delta_scales(self, global_values: torch.Tensor, global_mask: torch.Tensor) -> torch.Tensor:
        configured_scales = self.phenotype_scales.to(global_values.device, global_values.dtype)
        observed_count = global_mask.float().sum(dim=0)
        safe_count = observed_count.clamp_min(1.0)
        mean = global_values.masked_fill(~global_mask, 0.0).sum(dim=0) / safe_count
        centered = (global_values - mean).masked_fill(~global_mask, 0.0)
        batch_scale = torch.sqrt(centered.pow(2).sum(dim=0) / safe_count).clamp_min(1e-6)
        return torch.where(configured_scales > 0, configured_scales, batch_scale)

    @staticmethod
    def _huber(error: torch.Tensor, delta: float) -> torch.Tensor:
        abs_error = error.abs()
        return torch.where(abs_error <= delta, 0.5 * error.pow(2), delta * (abs_error - 0.5 * delta))

    def forward(
        self,
        phenotype_values,
        phenotype_mask,
        labels=None,
        **table_inputs,
    ):
        local_embeddings = F.normalize(self.encode_table(**table_inputs), dim=-1)
        global_embeddings = all_gather_with_grad(local_embeddings)
        global_values = all_gather_tensor(phenotype_values.to(local_embeddings.device, local_embeddings.dtype))
        global_mask = all_gather_tensor(phenotype_mask.to(local_embeddings.device).bool())
        local_values = phenotype_values.to(local_embeddings.device, local_embeddings.dtype)
        local_mask = phenotype_mask.to(local_embeddings.device).bool()

        relations = self.relation_vectors(local_embeddings.dtype, local_embeddings.device)
        delta_embeddings = global_embeddings.unsqueeze(0) - local_embeddings.unsqueeze(1)
        pred_delta = torch.einsum("bgd,qd->bgq", delta_embeddings, relations)

        scales = self._delta_scales(global_values, global_mask)
        true_delta = (global_values.unsqueeze(0) - local_values.unsqueeze(1)) / scales.view(1, 1, -1)
        pair_mask = local_mask.unsqueeze(1) & global_mask.unsqueeze(0)
        if self.min_pair_delta > 0:
            pair_mask = pair_mask & (true_delta.abs() >= self.min_pair_delta)

        local_batch_size = local_embeddings.size(0)
        start = gather_batch_start(local_batch_size, local_embeddings.device)
        row_indices = torch.arange(local_batch_size, device=local_embeddings.device)
        self_mask = torch.zeros(
            local_batch_size,
            global_embeddings.size(0),
            dtype=torch.bool,
            device=local_embeddings.device,
        )
        self_mask[row_indices, start + row_indices] = True
        pair_mask = pair_mask & (~self_mask.unsqueeze(-1))

        pair_count = pair_mask.float().sum()
        if pair_count <= 0:
            zero = local_embeddings.sum() * 0.0 + relations.sum() * 0.0
            return zero, {
                "loss_sum": zero.detach(),
                "abs_error_sum": zero.detach(),
                "squared_error_sum": zero.detach(),
                "pair_count": zero.detach(),
            }

        projection_error = pred_delta - true_delta
        projection_terms = self._huber(projection_error, self.huber_delta)
        projection_loss_sum = projection_terms[pair_mask].sum()
        projection_loss = projection_loss_sum / pair_count

        loss = self.projection_loss_weight * projection_loss
        loss_sum_for_logging = self.projection_loss_weight * projection_loss_sum

        if self.transe_loss_weight > 0:
            transe_target = true_delta.unsqueeze(-1) * relations.view(1, 1, relations.size(0), relations.size(1))
            transe_error = delta_embeddings.unsqueeze(2) - transe_target
            transe_terms = transe_error.pow(2).mean(dim=-1)
            transe_loss_sum = transe_terms[pair_mask].sum()
            transe_loss = transe_loss_sum / pair_count
            loss = loss + self.transe_loss_weight * transe_loss
            loss_sum_for_logging = loss_sum_for_logging + self.transe_loss_weight * transe_loss_sum

        if self.relation_l2_weight > 0:
            loss = loss + self.relation_l2_weight * relations.pow(2).mean()

        abs_error_sum = projection_error[pair_mask].abs().sum()
        squared_error_sum = projection_error[pair_mask].pow(2).sum()
        return loss, {
            "loss_sum": loss_sum_for_logging.detach(),
            "abs_error_sum": abs_error_sum.detach(),
            "squared_error_sum": squared_error_sum.detach(),
            "pair_count": pair_count.detach(),
        }


__all__ = [
    "AttentionPooling",
    "PhenotypeMetricModel",
    "all_gather_tensor",
    "all_gather_with_grad",
    "gather_batch_start",
    "is_distributed",
]
