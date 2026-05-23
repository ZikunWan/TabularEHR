import builtins
import hashlib
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional


def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0
    local_rank = os.environ.get("LOCAL_RANK")
    return local_rank is None or int(local_rank) in (-1, 0)


def _configure_non_main_process_logging() -> None:
    if _is_main_process():
        return

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("ACCELERATE_LOG_LEVEL", "error")
    logging.basicConfig(level=logging.ERROR, force=True)
    logging.getLogger().setLevel(logging.ERROR)
    for logger_name in ("transformers", "accelerate", "deepspeed", "torch", "torch.distributed"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


_configure_non_main_process_logging()

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import Dataset, get_worker_info
from tqdm.auto import tqdm
from transformers import EarlyStoppingCallback, HfArgumentParser, PreTrainedModel, Trainer, TrainingArguments

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.mimic.mimic_dataset import MIMICIV
from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D
from utils.collate import build_table_token_tensors
from utils.load_embedding import build_text_to_idx
from utils.weight_loader import load_model_weights


def is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    rank = os.environ.get("RANK")
    return rank is None or int(rank) == 0


def rank0_print(*args, **kwargs):
    if is_rank0():
        builtins.print(*args, **kwargs)


print = rank0_print


def build_state_key(dataset_name: str, sample_info: dict[str, Any]) -> str:
    if dataset_name == "mimic_iv":
        return (
            f"mimic_iv|{sample_info.get('subject_id', '')}|"
            f"{sample_info.get('task', '')}|"
            f"{sample_info.get('context_begin', '')}|"
            f"{sample_info.get('context_end', '')}"
        )
    if dataset_name == "eicu":
        return (
            f"eicu|{sample_info.get('patient_id', '')}|"
            f"{sample_info.get('icustay_id', '')}|"
            f"{sample_info.get('task_name', '')}|"
            f"{sample_info.get('obs_hours', '')}|"
            f"{sample_info.get('gap_hours', '')}|"
            f"{sample_info.get('pred_hours', '')}"
        )
    if dataset_name == "ehrshot":
        return (
            f"ehrshot|{sample_info.get('patient_id', '')}|"
            f"{sample_info.get('task_name', '')}|"
            f"{sample_info.get('prediction_time', '')}"
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def deserialize_table_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(records)
    if table.empty:
        return pd.DataFrame(columns=["Time", "Item", "Value", "Unit", "Category"])

    preferred_columns = ["Time", "Item", "Value", "Unit", "Category"]
    table = table[[c for c in preferred_columns if c in table.columns]]
    if "Time" in table.columns:
        table["Time"] = pd.to_datetime(table["Time"], errors="coerce")
    return table


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
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            grad_stack = torch.stack([grad.contiguous() for grad in grads], dim=0)
            dist.all_reduce(grad_stack, op=dist.ReduceOp.SUM)
            return grad_stack[ctx.rank]
        return grads[ctx.rank]


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return tensor

    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
    size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, local_size)
    sizes = [int(s.item()) for s in size_list]
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


def all_gather_subject_ids(subject_ids: Optional[List[str]]) -> Optional[List[str]]:
    if subject_ids is None:
        return None
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return list(subject_ids)

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, list(subject_ids))
    return [subject_id for rank_subject_ids in gathered for subject_id in rank_subject_ids]


@dataclass
class DataArguments:
    dataset: List[str] = field(default_factory=lambda: ["mimic_iv"])
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/next_token_prediction.csv"
    )
    val_sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val/next_token_prediction.csv"
    )
    table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings.pt"]
    )
    markdown_embedding_path: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/mimic-iv-3.1_tabular/embeddings/table_free_text/embeddings.pt"]
    )
    eicu_root_dir: str = field(default="/data/zikun_workspace/eicu-crd")
    eicu_processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    eicu_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json")
    eicu_val_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json")
    eicu_table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt"]
    )
    eicu_markdown_embedding_path: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/eicu-crd/processed/pretraining_markdown_embeddings/embeddings.pt"]
    )
    ehrshot_root_dir: str = field(default="/data/EHR_data_public/EHRSHOT")
    ehrshot_sample_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/pretraining_index/sample_info_train.csv")
    ehrshot_val_sample_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/pretraining_index/sample_info_val.csv")
    ehrshot_table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt"]
    )
    ehrshot_markdown_embedding_path: List[str] = field(
        default_factory=lambda: ["/data/EHR_data_public/EHRSHOT/pretraining_markdown_embeddings/embeddings.pt"]
    )
    markdown_candidate_count: int = field(default=1024)
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    pretrained_path: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_table_len: Optional[int] = field(default=4096)
    min_table_rows: int = field(default=2)
    table_loss_weight: float = field(default=1.0)
    markdown_loss_weight: float = field(default=1.0)
    augmentation_seed: int = field(default=42)
    temperature: float = field(default=0.07)


@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    output_dir: str = field(default="/data/zikun_workspace/checkpoints/pretraining/contrastive_learning")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=64)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-5)
    warmup_steps: int = field(default=100)
    weight_decay: float = field(default=0.01)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    save_total_limit: int = field(default=1)
    bf16: bool = field(default=True)
    dataloader_num_workers: int = field(default=32)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="wandb")
    wandb_project: Optional[str] = field(default="Contrastive_Learning")
    metric_for_best_model: str = field(default="eval_recall@1")
    greater_is_better: bool = field(default=True)
    early_stopping_patience: int = field(default=10)
    eval_retrieval_device: str = field(default="cpu")
    eval_retrieval_chunk_size: int = field(default=1024)

    def __post_init__(self):
        super().__post_init__()
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        self.eval_strategy = "steps"
        self.load_best_model_at_end = True
        self.greater_is_better = True


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        scores = self.attention(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class ContrastiveModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config: LongTableEncoder1DConfig,
        embedding_matrix: torch.Tensor,
        markdown_embedding_matrix: torch.Tensor,
        temperature: float = 0.07,
        table_loss_weight: float = 1.0,
        markdown_loss_weight: float = 1.0,
    ):
        super().__init__(config)
        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.temperature = temperature
        self.table_loss_weight = float(table_loss_weight)
        self.markdown_loss_weight = float(markdown_loss_weight)
        self.text_embedding_matrix = embedding_matrix.cpu()
        self.markdown_embedding_matrix = markdown_embedding_matrix.cpu()
        pool_hidden_size = config.dim_out if config.dim_out is not None else config.dim
        self.table_pooling = AttentionPooling(pool_hidden_size)

    def lookup_text_embeddings(self, token_ids: torch.Tensor, dtype: torch.dtype, device: torch.device):
        original_shape = token_ids.shape
        selected = self.text_embedding_matrix.index_select(0, token_ids.reshape(-1).cpu())
        selected = selected.to(device=device, dtype=dtype, non_blocking=True)
        return selected.view(*original_shape, selected.shape[-1])

    def lookup_markdown_embeddings(self, candidate_indices: torch.Tensor, dtype: torch.dtype, device: torch.device):
        original_shape = candidate_indices.shape
        selected = self.markdown_embedding_matrix.index_select(0, candidate_indices.reshape(-1).cpu())
        selected = selected.to(device=device, dtype=dtype, non_blocking=True)
        return selected.view(*original_shape, selected.shape[-1])

    def encode_table(
        self,
        item_ids,
        unit_ids,
        value_text_ids,
        times,
        numeric_values,
        numeric_mask,
        seq_mask=None,
        type_ids=None,
    ):
        embedding_dtype = self.encoder.embedding.item_proj.weight.dtype
        embedding_device = self.encoder.embedding.item_proj.weight.device
        hidden_states, hidden_mask = self.encoder(
            item_emb=self.lookup_text_embeddings(item_ids, embedding_dtype, embedding_device),
            unit_emb=self.lookup_text_embeddings(unit_ids, embedding_dtype, embedding_device),
            value_emb=self.lookup_text_embeddings(value_text_ids, embedding_dtype, embedding_device),
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            seq_mask=seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )
        return self.table_pooling(self.adapter(hidden_states, hidden_mask))

    def _mask_same_subject(self, logits: torch.Tensor, subject_ids: Optional[List[str]]):
        if subject_ids is None:
            return logits
        mask = torch.tensor(
            [[subject_id == other_subject_id for other_subject_id in subject_ids] for subject_id in subject_ids],
            dtype=torch.bool,
            device=logits.device,
        )
        row_indices = torch.arange(logits.size(0), device=logits.device)
        mask[row_indices, row_indices] = False
        return logits.masked_fill(mask, float("-inf"))

    def _table_loss(self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor, subject_ids: Optional[List[str]]):
        all_anchor_emb = all_gather_with_grad(anchor_emb)
        all_positive_emb = all_gather_with_grad(positive_emb)
        all_subject_ids = all_gather_subject_ids(subject_ids)
        labels = torch.arange(all_anchor_emb.size(0), device=anchor_emb.device)

        a2p_logits = torch.matmul(all_anchor_emb, all_positive_emb.t()) / self.temperature
        p2a_logits = torch.matmul(all_positive_emb, all_anchor_emb.t()) / self.temperature
        a2p_logits = self._mask_same_subject(a2p_logits, all_subject_ids)
        p2a_logits = self._mask_same_subject(p2a_logits, all_subject_ids)

        loss = 0.5 * (F.cross_entropy(a2p_logits, labels) + F.cross_entropy(p2a_logits, labels))
        with torch.no_grad():
            recall = 0.5 * (
                (a2p_logits.argmax(dim=1) == labels).float().mean()
                + (p2a_logits.argmax(dim=1) == labels).float().mean()
            )
        return loss, recall

    def forward(
        self,
        positive_,
        markdown_candidate_indices,
        markdown_labels,
        subject_ids=None,
        **anchor_inputs,
    ):
        anchor_emb = F.normalize(self.encode_table(**anchor_inputs), dim=-1)
        positive_emb = F.normalize(self.encode_table(**positive_), dim=-1)
        table_loss, table_recall1 = self._table_loss(anchor_emb, positive_emb, subject_ids)

        markdown_emb = self.lookup_markdown_embeddings(
            markdown_candidate_indices,
            dtype=anchor_emb.dtype,
            device=anchor_emb.device,
        )
        markdown_emb = F.normalize(markdown_emb.to(anchor_emb.dtype), dim=-1)
        markdown_logits = torch.einsum("bd,bkd->bk", anchor_emb, markdown_emb) / self.temperature
        markdown_labels = markdown_labels.to(markdown_logits.device)
        markdown_loss = F.cross_entropy(markdown_logits, markdown_labels)

        with torch.no_grad():
            markdown_positive_emb = markdown_emb[:, 0, :]
            topk_indices = markdown_logits.argsort(dim=1, descending=True)
            markdown_metrics = {}
            for k in (1, 5, 10, 50):
                if k <= markdown_logits.size(1):
                    hits = (topk_indices[:, :k] == markdown_labels.unsqueeze(1)).any(dim=1).float().mean()
                    markdown_metrics[f"markdown_recall@{k}"] = hits.detach()

        loss_weight = self.table_loss_weight + self.markdown_loss_weight
        if loss_weight <= 0:
            raise ValueError("At least one contrastive loss weight must be positive.")
        weighted_loss = (
            self.table_loss_weight * table_loss
            + self.markdown_loss_weight * markdown_loss
        ) / loss_weight

        outputs = {
            "anchor_embs": anchor_emb,
            "positive_embs": positive_emb,
            "markdown_positive_embs": markdown_positive_emb,
            "subject_ids": subject_ids,
            "table_loss": table_loss.detach(),
            "markdown_loss": markdown_loss.detach(),
            "table_recall@1": table_recall1.detach(),
        }
        outputs.update(markdown_metrics)
        return weighted_loss, outputs


class PatientStateDataset(Dataset):
    def __init__(self, sample_keys: List[str], state_tables: dict[str, dict[str, Any]], is_eval: bool = False):
        self.sample_keys = sample_keys
        self.state_tables = state_tables
        self.is_eval = is_eval

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        sample_key = self.sample_keys[idx]
        record = self.state_tables[sample_key]
        return {
            "sample_key": sample_key,
            "subject_id": str(record["subject_id"]),
            "anchor_table_records": record["anchor_table_records"],
            "positive_table_records": record["positive_table_records"],
            "is_eval": self.is_eval,
        }


class ContrastiveDataCollator:
    def __init__(
        self,
        text_to_idx: dict[str, int],
        pad_idx: int,
        type_vocab: dict[str, int],
        max_table_len: Optional[int],
        min_table_rows: int,
        markdown_key_to_idx: dict[str, int],
        markdown_keys: List[str],
        markdown_subject_ids: List[str],
        markdown_candidate_count: int,
        augmentation_seed: int,
    ):
        if markdown_candidate_count < 2:
            raise ValueError("markdown_candidate_count must be at least 2.")
        self.text_to_idx = text_to_idx
        self.pad_idx = pad_idx
        self.type_vocab = type_vocab
        self.max_table_len = max_table_len
        self.min_table_rows = min_table_rows
        self.markdown_key_to_idx = markdown_key_to_idx
        self.markdown_keys = markdown_keys
        self.markdown_subject_ids = markdown_subject_ids
        self.markdown_candidate_count = markdown_candidate_count
        self.augmentation_seed = augmentation_seed
        self._batch_counter = 0

    def _sample_markdown_candidates(self, sample_key: str, subject_id: str, rng: random.Random):
        positive_idx = self.markdown_key_to_idx[sample_key]
        candidate_indices = [positive_idx]
        seen = {positive_idx}

        attempts = 0
        max_attempts = self.markdown_candidate_count * 20 + 100
        while len(candidate_indices) < self.markdown_candidate_count and attempts < max_attempts:
            attempts += 1
            candidate_idx = rng.randrange(len(self.markdown_keys))
            if candidate_idx in seen:
                continue
            if self.markdown_subject_ids[candidate_idx] == subject_id:
                continue
            candidate_indices.append(candidate_idx)
            seen.add(candidate_idx)

        if len(candidate_indices) < self.markdown_candidate_count:
            for candidate_idx, candidate_subject_id in enumerate(self.markdown_subject_ids):
                if candidate_idx in seen or candidate_subject_id == subject_id:
                    continue
                candidate_indices.append(candidate_idx)
                seen.add(candidate_idx)
                if len(candidate_indices) == self.markdown_candidate_count:
                    break

        if len(candidate_indices) < self.markdown_candidate_count:
            raise ValueError(
                f"Only found {len(candidate_indices)} markdown candidates for subject_id={subject_id}; "
                f"required {self.markdown_candidate_count}."
            )
        return torch.tensor(candidate_indices, dtype=torch.long)

    def __call__(self, batch):
        anchor_tables = []
        positive_tables = []
        markdown_candidate_indices = []
        subject_ids = []
        batch_counter = self._batch_counter
        self._batch_counter += 1
        worker_info = get_worker_info()
        worker_seed = int(worker_info.seed) if worker_info is not None else 0

        for item in batch:
            sample_key = item["sample_key"]
            anchor_table = deserialize_table_records(item["anchor_table_records"])
            positive_table = deserialize_table_records(item["positive_table_records"])
            if self.max_table_len is not None:
                anchor_table = anchor_table.tail(self.max_table_len).reset_index(drop=True)
                positive_table = positive_table.tail(self.max_table_len).reset_index(drop=True)
            if len(anchor_table) < self.min_table_rows or len(positive_table) < self.min_table_rows:
                continue

            if item.get("is_eval", False):
                rng_seed = stable_seed(sample_key, self.augmentation_seed)
            else:
                rng_seed = stable_seed(f"{sample_key}|{worker_seed}|{batch_counter}", self.augmentation_seed)
            rng = random.Random(rng_seed)

            anchor_tables.append(anchor_table)
            positive_tables.append(positive_table)
            subject_ids.append(item["subject_id"])
            markdown_candidate_indices.append(
                self._sample_markdown_candidates(sample_key, item["subject_id"], rng)
            )

        if len(anchor_tables) == 0:
            raise ValueError("All samples in this batch are invalid after cached table filtering.")

        anchor_tensors = build_table_token_tensors(
            anchor_tables,
            text_to_idx=self.text_to_idx,
            pad_idx=self.pad_idx,
            type_vocab=self.type_vocab,
        )
        positive_tensors = build_table_token_tensors(
            positive_tables,
            text_to_idx=self.text_to_idx,
            pad_idx=self.pad_idx,
            type_vocab=self.type_vocab,
        )

        return {
            **anchor_tensors,
            "positive_": positive_tensors,
            "markdown_candidate_indices": torch.stack(markdown_candidate_indices),
            "markdown_labels": torch.zeros(len(anchor_tables), dtype=torch.long),
            "subject_ids": subject_ids,
            "labels": torch.zeros(len(anchor_tables), dtype=torch.long),
        }


class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("labels", None)
        loss, outputs = model(**inputs)
        if return_outputs:
            return loss, outputs
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        model = self.model
        model.eval()
        anchor_embs = []
        positive_embs = []
        markdown_embs = []
        subject_ids = []
        total_loss = 0.0
        num_batches = 0

        for inputs in tqdm(
            eval_dataloader,
            desc=f"Eval forward step {self.state.global_step}",
            disable=not is_rank0(),
            dynamic_ncols=True,
            leave=False,
        ):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            total_loss += loss.mean().item()
            num_batches += 1
            anchor_embs.append(outputs["anchor_embs"].detach().float().cpu())
            positive_embs.append(outputs["positive_embs"].detach().float().cpu())
            markdown_embs.append(outputs["markdown_positive_embs"].detach().float().cpu())
            subject_ids.extend(inputs["subject_ids"])

        model.train()

        local_anchor = torch.cat(anchor_embs, dim=0)
        local_positive = torch.cat(positive_embs, dim=0)
        local_markdown = torch.cat(markdown_embs, dim=0)
        local_subject_ids = subject_ids
        distributed_eval = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed_eval else 0

        if distributed_eval:
            query_counts = [None] * dist.get_world_size()
            anchor_list = [None] * dist.get_world_size()
            positive_list = [None] * dist.get_world_size()
            markdown_list = [None] * dist.get_world_size()
            subject_id_list = [None] * dist.get_world_size()
            dist.all_gather_object(query_counts, local_anchor.shape[0])
            dist.all_gather_object(anchor_list, local_anchor)
            dist.all_gather_object(positive_list, local_positive)
            dist.all_gather_object(markdown_list, local_markdown)
            dist.all_gather_object(subject_id_list, local_subject_ids)
            query_counts = [int(count) for count in query_counts]
            query_offset = sum(query_counts[:rank])
            global_query_count = sum(query_counts)
            anchor_all = torch.cat(anchor_list, dim=0)
            positive_all = torch.cat(positive_list, dim=0)
            markdown_all = torch.cat(markdown_list, dim=0)
            subject_ids_all = [sid for rank_sids in subject_id_list for sid in rank_sids]
        else:
            query_offset = 0
            global_query_count = local_anchor.shape[0]
            anchor_all = local_anchor
            positive_all = local_positive
            markdown_all = local_markdown
            subject_ids_all = local_subject_ids

        loss_tensor = torch.tensor(total_loss / max(num_batches, 1))
        if distributed_eval:
            loss_tensor = loss_tensor.to(self.args.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_tensor = loss_tensor.cpu()

        retrieval_device = self.args.eval_retrieval_device
        if retrieval_device in {"cuda", "auto"} and getattr(self.args, "device", None) is not None:
            retrieval_device = str(self.args.device)

        positive_indices = [[row_idx] for row_idx in range(query_offset, query_offset + local_anchor.shape[0])]
        table_forward = compute_recall_metrics(
            local_anchor,
            positive_all,
            query_subject_ids=local_subject_ids,
            positive_subject_ids=subject_ids_all,
            positive_indices=positive_indices,
            compute_device=retrieval_device,
            chunk_size=self.args.eval_retrieval_chunk_size,
            return_sums=True,
            progress_desc=f"Eval table retrieval step {self.state.global_step}",
            show_progress=is_rank0(),
        )
        table_reverse = compute_recall_metrics(
            local_positive,
            anchor_all,
            query_subject_ids=local_subject_ids,
            positive_subject_ids=subject_ids_all,
            positive_indices=positive_indices,
            compute_device=retrieval_device,
            chunk_size=self.args.eval_retrieval_chunk_size,
            return_sums=True,
            progress_desc=f"Eval table reverse retrieval step {self.state.global_step}",
            show_progress=is_rank0(),
        )
        markdown_metrics = compute_recall_metrics(
            local_anchor,
            markdown_all,
            query_subject_ids=local_subject_ids,
            positive_subject_ids=subject_ids_all,
            positive_indices=positive_indices,
            compute_device=retrieval_device,
            chunk_size=self.args.eval_retrieval_chunk_size,
            return_sums=True,
            progress_desc=f"Eval markdown retrieval step {self.state.global_step}",
            show_progress=is_rank0(),
        )

        table_keys = sorted(k for k in table_forward if k.startswith("recall@"))
        markdown_keys = sorted(k for k in markdown_metrics if k.startswith("recall@"))
        table_tensor = torch.tensor(
            [
                table_forward["num_queries"] + table_reverse["num_queries"],
                table_forward["mrr"] + table_reverse["mrr"],
                *[table_forward[key] + table_reverse[key] for key in table_keys],
            ],
            dtype=torch.float64,
            device=self.args.device,
        )
        markdown_tensor = torch.tensor(
            [
                markdown_metrics["num_queries"],
                markdown_metrics["mrr"],
                *[markdown_metrics[key] for key in markdown_keys],
            ],
            dtype=torch.float64,
            device=self.args.device,
        )
        if distributed_eval:
            dist.all_reduce(table_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(markdown_tensor, op=dist.ReduceOp.SUM)

        metrics = {f"{metric_key_prefix}_loss": loss_tensor.item()}
        table_queries = max(table_tensor[0].item(), 1.0)
        metrics[f"{metric_key_prefix}_table_mrr"] = table_tensor[1].item() / table_queries
        for idx, key in enumerate(table_keys, start=2):
            metrics[f"{metric_key_prefix}_table_{key}"] = table_tensor[idx].item() / table_queries

        markdown_queries = max(markdown_tensor[0].item(), 1.0)
        metrics[f"{metric_key_prefix}_markdown_mrr"] = markdown_tensor[1].item() / markdown_queries
        for idx, key in enumerate(markdown_keys, start=2):
            metrics[f"{metric_key_prefix}_markdown_{key}"] = markdown_tensor[idx].item() / markdown_queries

        table_recall1 = metrics.get(f"{metric_key_prefix}_table_recall@1", 0.0)
        markdown_recall1 = metrics.get(f"{metric_key_prefix}_markdown_recall@1", 0.0)
        metrics[f"{metric_key_prefix}_recall@1"] = 0.5 * (table_recall1 + markdown_recall1)

        if rank == 0:
            print(f"[Eval] step={self.state.global_step} N={global_query_count} " + " ".join(
                f"{key}={value:.4f}" for key, value in metrics.items()
            ))

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


def compute_recall_metrics(
    anchor_embs: torch.Tensor,
    positive_embs: torch.Tensor,
    query_subject_ids=None,
    positive_subject_ids=None,
    positive_indices=None,
    chunk_size: int = 1024,
    show_progress: bool = False,
    progress_desc: str = "Eval retrieval",
    compute_device: Optional[str] = None,
    return_sums: bool = False,
):
    num_queries = anchor_embs.size(0)
    num_positives = positive_embs.size(0)
    if positive_indices is None:
        positive_indices = [[row_idx] for row_idx in range(num_queries)]

    recall_hits = {k: 0.0 for k in [1, 5, 10, 50] if k <= num_positives}
    max_k = max(recall_hits.keys(), default=1)
    reciprocal_rank_sum = 0.0
    if compute_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif compute_device is not None:
        device = torch.device(compute_device)
    else:
        device = anchor_embs.device

    positive_index_tensor = torch.tensor(
        [indices[0] if indices else -1 for indices in positive_indices],
        dtype=torch.long,
        device=device,
    )

    query_subject_tensor = None
    positive_subject_tensor = None
    if query_subject_ids is not None and positive_subject_ids is not None:
        subject_to_id = {}
        query_subject_values = []
        positive_subject_values = []
        for subject_id in query_subject_ids:
            subject_to_id.setdefault(subject_id, len(subject_to_id))
            query_subject_values.append(subject_to_id[subject_id])
        for subject_id in positive_subject_ids:
            subject_to_id.setdefault(subject_id, len(subject_to_id))
            positive_subject_values.append(subject_to_id[subject_id])
        query_subject_tensor = torch.tensor(query_subject_values, dtype=torch.long, device=device)
        positive_subject_tensor = torch.tensor(positive_subject_values, dtype=torch.long, device=device)

    positive_embs_t = positive_embs.to(device=device, non_blocking=True).t()
    chunk_starts = tqdm(
        range(0, num_queries, chunk_size),
        total=(num_queries + chunk_size - 1) // chunk_size,
        desc=progress_desc,
        disable=not show_progress,
        dynamic_ncols=True,
        leave=False,
    )
    for start in chunk_starts:
        end = min(start + chunk_size, num_queries)
        sim = torch.matmul(anchor_embs[start:end].to(device=device, non_blocking=True), positive_embs_t)
        chunk_positive_indices = positive_index_tensor[start:end]

        if query_subject_tensor is not None and positive_subject_tensor is not None:
            same_subject_mask = query_subject_tensor[start:end, None] == positive_subject_tensor[None, :]
            same_subject_mask[torch.arange(end - start, device=device), chunk_positive_indices] = False
            sim = sim.masked_fill(same_subject_mask, float("-inf"))

        topk_indices = torch.topk(sim, k=max_k, dim=1).indices
        hits = topk_indices == chunk_positive_indices[:, None]
        for k in recall_hits:
            recall_hits[k] += hits[:, :k].any(dim=1).float().sum().item()

        positive_scores = sim.gather(1, chunk_positive_indices[:, None]).squeeze(1)
        ranks = (sim > positive_scores[:, None]).sum(dim=1).float() + 1.0
        reciprocal_rank_sum += (1.0 / ranks).sum().item()

    metrics = {}
    for k, hits in recall_hits.items():
        metrics[f"recall@{k}"] = hits if return_sums else hits / max(num_queries, 1)
    metrics["mrr"] = reciprocal_rank_sum if return_sums else reciprocal_rank_sum / max(num_queries, 1)
    if return_sums:
        metrics["num_queries"] = float(num_queries)
    return metrics


def load_table_embeddings(cache_paths: List[str]):
    embedding_cache = {}
    text_dim = None
    for cache_path in cache_paths:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        embedding_cache.update(data["embeddings"])
        text_dim = int(data["text_dim"])
        print(f"Loaded {len(data['embeddings'])} embeddings from {cache_path}")
    if text_dim is None:
        raise ValueError("No table text embeddings loaded.")

    vocab_keys = list(embedding_cache.keys())
    text_to_idx = build_text_to_idx(vocab_keys)
    matrix = torch.empty(len(vocab_keys), text_dim)
    for idx, text in enumerate(vocab_keys):
        matrix[idx] = embedding_cache[text]
    return text_dim, vocab_keys, text_to_idx, matrix


def load_patient_state_cache(paths: List[str], expected_cache_config: Optional[dict[str, Any]] = None):
    markdown_embeddings = {}
    state_tables = {}
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        state_table_path = os.path.join(os.path.dirname(path), "state_tables.pt")
        metadata_path = os.path.join(os.path.dirname(path), "metadata.json")
        if not os.path.exists(state_table_path):
            raise FileNotFoundError(
                f"Missing patient-state table cache: {state_table_path}. "
                f"Rebuild the cache with preprocess/generate_markdown_embeddings.py."
            )
        if expected_cache_config is not None and os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            for key, expected_value in expected_cache_config.items():
                if metadata.get(key) != expected_value:
                    raise ValueError(
                        f"Markdown cache config mismatch for {path}: metadata[{key}]={metadata.get(key)!r} "
                        f"but expected {expected_value!r}. Rebuild the cache or update training args."
                    )
        table_records = torch.load(state_table_path, map_location="cpu", weights_only=False)
        markdown_embeddings.update(data)
        state_tables.update(table_records)
        print(f"Loaded {len(data)} markdown embeddings from {path}")
        print(f"Loaded {len(table_records)} patient-state tables from {state_table_path}")

    markdown_keys = [key for key in markdown_embeddings if key in state_tables]
    if len(markdown_keys) == 0:
        raise ValueError("No markdown embeddings with matching patient-state tables were loaded.")

    first_embedding = markdown_embeddings[markdown_keys[0]]
    markdown_embedding_matrix = torch.empty(len(markdown_keys), first_embedding.numel(), dtype=first_embedding.dtype)
    for idx, key in enumerate(markdown_keys):
        markdown_embedding_matrix[idx] = markdown_embeddings[key].reshape(-1)

    markdown_key_to_idx = {key: idx for idx, key in enumerate(markdown_keys)}
    markdown_subject_ids = [str(state_tables[key]["subject_id"]) for key in markdown_keys]
    return markdown_key_to_idx, markdown_keys, markdown_subject_ids, markdown_embedding_matrix, state_tables


def get_embedding_cache_paths(data_args: DataArguments):
    paths = []
    for dataset_name in data_args.dataset:
        if dataset_name == "mimic_iv":
            paths.extend(data_args.table_text_embedding)
        elif dataset_name == "eicu":
            paths.extend(data_args.eicu_table_text_embedding)
        elif dataset_name == "ehrshot":
            paths.extend(data_args.ehrshot_table_text_embedding)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return paths


def get_markdown_embedding_paths(data_args: DataArguments):
    paths = []
    for dataset_name in data_args.dataset:
        if dataset_name == "mimic_iv":
            paths.extend(data_args.markdown_embedding_path)
        elif dataset_name == "eicu":
            paths.extend(data_args.eicu_markdown_embedding_path)
        elif dataset_name == "ehrshot":
            paths.extend(data_args.ehrshot_markdown_embedding_path)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return paths


def build_one_dataset(dataset_name: str, data_args: DataArguments, sample_info_path: str):
    if dataset_name == "mimic_iv":
        return MIMICIV(
            root_dir=data_args.root_dir,
            sample_info_path=sample_info_path,
            lazy_mode=True,
            shuffle=False,
            table_mode="table_only",
            max_samples=None,
            use_table_length_cache=False,
        )
    if dataset_name == "eicu":
        return EICUDataset(
            root_dir=data_args.eicu_root_dir,
            processed_dir=data_args.eicu_processed_dir,
            sample_info_path=sample_info_path,
            task_name=None,
            lazy_mode=True,
            shuffle=False,
            table_mode="table_only",
        )
    if dataset_name == "ehrshot":
        return EHRSHOTDataset(
            root_dir=data_args.ehrshot_root_dir,
            sample_info_path=sample_info_path,
            task_name=None,
            lazy_mode=True,
            table_mode="table_only",
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def split_sample_info_path(data_args: DataArguments, dataset_name: str, split: str):
    if dataset_name == "mimic_iv":
        return data_args.sample_info_path if split == "train" else data_args.val_sample_info_path
    if dataset_name == "eicu":
        return data_args.eicu_sample_info_path if split == "train" else data_args.eicu_val_sample_info_path
    if dataset_name == "ehrshot":
        return data_args.ehrshot_sample_info_path if split == "train" else data_args.ehrshot_val_sample_info_path
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_split_keys(data_args: DataArguments, split: str, state_tables: dict[str, Any], markdown_key_to_idx):
    keys = []
    max_samples = data_args.max_train_samples if split == "train" else data_args.max_eval_samples
    for dataset_name in data_args.dataset:
        dataset = build_one_dataset(dataset_name, data_args, split_sample_info_path(data_args, dataset_name, split))
        dataset_keys = []
        for sample_info in dataset.sample_info:
            sample_key = build_state_key(dataset_name, sample_info)
            if sample_key in state_tables and sample_key in markdown_key_to_idx:
                dataset_keys.append(sample_key)
        if max_samples is not None:
            dataset_keys = dataset_keys[:max_samples]
        print(f"{split} {dataset_name} samples: {len(dataset_keys)}")
        keys.extend(dataset_keys)
    return keys


def main():
    parser = HfArgumentParser((DataArguments, TrainingArgumentsCustom))
    data_args, training_args = parser.parse_args_into_dataclasses()
    expected_cache_config = {
        "cache_schema_version": 4,
        "max_table_len": data_args.max_table_len,
        "min_table_rows": data_args.min_table_rows,
        "view_strategy": "patient_state_category_views",
        "markdown_text": "full_patient_state",
    }

    print("Stage 2: Patient-State Contrastive Learning")
    print(f"Datasets: {', '.join(data_args.dataset)}")
    print(f"Table text embeddings: {get_embedding_cache_paths(data_args)}")
    print(f"Markdown embeddings: {get_markdown_embedding_paths(data_args)}")
    print(f"Markdown candidate count: {data_args.markdown_candidate_count}")
    print(f"Pretrained path: {data_args.pretrained_path}")
    print(f"Table loss weight: {data_args.table_loss_weight}")
    print(f"Markdown loss weight: {data_args.markdown_loss_weight}")

    text_dim, vocab_keys, text_to_idx, embedding_matrix = load_table_embeddings(get_embedding_cache_paths(data_args))
    (
        markdown_key_to_idx,
        markdown_keys,
        markdown_subject_ids,
        markdown_embedding_matrix,
        state_tables,
    ) = load_patient_state_cache(get_markdown_embedding_paths(data_args), expected_cache_config)

    with open(data_args.type_vocab_file, "r", encoding="utf-8") as f:
        type_vocab = {str(k): int(v) for k, v in json.load(f).items()}
    type_vocab_size = max(type_vocab.values()) + 1

    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=type_vocab_size,
        max_table_len=data_args.max_table_len,
        dim_out=2048,
    )
    model = ContrastiveModel(
        config=config,
        embedding_matrix=embedding_matrix,
        markdown_embedding_matrix=markdown_embedding_matrix,
        temperature=data_args.temperature,
        table_loss_weight=data_args.table_loss_weight,
        markdown_loss_weight=data_args.markdown_loss_weight,
    )
    model = load_model_weights(model, data_args.pretrained_path)

    train_keys = build_split_keys(data_args, "train", state_tables, markdown_key_to_idx)
    eval_keys = build_split_keys(data_args, "eval", state_tables, markdown_key_to_idx)
    if len(train_keys) == 0:
        raise ValueError("No training samples left after patient-state cache filtering.")
    if len(eval_keys) == 0:
        raise ValueError("No eval samples left after patient-state cache filtering.")

    train_dataset = PatientStateDataset(train_keys, state_tables, is_eval=False)
    eval_dataset = PatientStateDataset(eval_keys, state_tables, is_eval=True)
    collator = ContrastiveDataCollator(
        text_to_idx=text_to_idx,
        pad_idx=0,
        type_vocab=type_vocab,
        max_table_len=data_args.max_table_len,
        min_table_rows=data_args.min_table_rows,
        markdown_key_to_idx=markdown_key_to_idx,
        markdown_keys=markdown_keys,
        markdown_subject_ids=markdown_subject_ids,
        markdown_candidate_count=data_args.markdown_candidate_count,
        augmentation_seed=data_args.augmentation_seed,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Table vocab size: {len(vocab_keys)}, text_dim={text_dim}, type_vocab_size={type_vocab_size}")
    del vocab_keys, embedding_matrix, markdown_embedding_matrix

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
            )
        ],
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

    tabular_save_path = os.path.join(training_args.output_dir, "tabular_encoder")
    model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    os.makedirs(tabular_save_path, exist_ok=True)
    encoder_state_dict = {
        key: value.detach().cpu()
        for key, value in model_to_save.encoder.state_dict().items()
    }
    save_file(encoder_state_dict, os.path.join(tabular_save_path, "model.safetensors"))
    print(f"Saved standalone encoder to {tabular_save_path}")


if __name__ == "__main__":
    main()
