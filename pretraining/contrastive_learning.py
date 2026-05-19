import builtins
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional


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
from torch.utils.data import Dataset
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
        return grads[ctx.rank]


def all_gather_with_grad(tensor: torch.Tensor):
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return tensor, 0

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    sizes = [int(s.item()) for s in size_list]
    max_size = max(sizes)
    label_offset = sum(sizes[:rank])

    if tensor.size(0) < max_size:
        padding = torch.zeros(
            max_size - tensor.size(0),
            *tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat([tensor, padding], dim=0)

    gathered = GatherWithGrad.apply(tensor)
    gathered = [gathered[i][:sizes[i]] for i in range(world_size)]
    return torch.cat(gathered, dim=0), label_offset


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
    max_table_len: Optional[int] = field(default=16384)
    min_table_rows: int = field(default=2)
    view_keep_ratio: float = field(default=0.40)
    max_view_overlap_ratio: float = field(default=0.10)
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
        markdown_embedding_matrix: Optional[torch.Tensor] = None,
        temperature: float = 0.07,
    ):
        super().__init__(config)
        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.temperature = temperature
        self.text_embedding_matrix = embedding_matrix.cpu()
        self.markdown_embedding_matrix = markdown_embedding_matrix.cpu() if markdown_embedding_matrix is not None else None
        pool_hidden_size = config.dim_out if config.dim_out is not None else config.dim
        self.table_pooling = AttentionPooling(pool_hidden_size)

    def lookup_text_embeddings(self, token_ids: torch.Tensor, dtype: torch.dtype, device: torch.device):
        original_shape = token_ids.shape
        flat_ids = token_ids.reshape(-1).cpu()
        selected = self.text_embedding_matrix.index_select(0, flat_ids)
        selected = selected.to(device=device, dtype=dtype, non_blocking=True)
        return selected.view(*original_shape, selected.shape[-1])

    def lookup_markdown_embeddings(self, markdown_candidate_indices: torch.Tensor, dtype: torch.dtype, device: torch.device):
        if self.markdown_embedding_matrix is None:
            raise ValueError("markdown_embedding_matrix is required when markdown_candidate_indices are provided.")
        original_shape = markdown_candidate_indices.shape
        flat_indices = markdown_candidate_indices.reshape(-1).cpu()
        selected = self.markdown_embedding_matrix.index_select(0, flat_indices)
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
        item_emb = self.lookup_text_embeddings(item_ids, dtype=embedding_dtype, device=embedding_device)
        unit_emb = self.lookup_text_embeddings(unit_ids, dtype=embedding_dtype, device=embedding_device)
        value_emb = self.lookup_text_embeddings(value_text_ids, dtype=embedding_dtype, device=embedding_device)
        hidden_states, hidden_mask = self.encoder(
            item_emb=item_emb,
            unit_emb=unit_emb,
            value_emb=value_emb,
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            seq_mask=seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )
        query_embeddings = self.adapter(hidden_states, hidden_mask)
        return self.table_pooling(query_embeddings)

    def forward(
        self,
        positive_,
        markdown_embeddings=None,
        markdown_candidate_indices=None,
        markdown_labels=None,
        subject_ids=None,
        **anchor_inputs,
    ):
        anchor_emb = F.normalize(self.encode_table(**anchor_inputs), dim=-1)
        positive_emb = F.normalize(self.encode_table(**positive_), dim=-1)

        all_positive_emb, label_offset = all_gather_with_grad(positive_emb)
        aug_logits = torch.matmul(anchor_emb, all_positive_emb.t()) / self.temperature
        labels = label_offset + torch.arange(anchor_emb.size(0), device=aug_logits.device)
        loss = F.cross_entropy(aug_logits, labels)

        with torch.no_grad():
            recall1 = (aug_logits.argmax(dim=1) == labels).float().mean()

        markdown_positive_emb = None
        if markdown_candidate_indices is not None:
            markdown_embeddings = self.lookup_markdown_embeddings(
                markdown_candidate_indices,
                dtype=anchor_emb.dtype,
                device=anchor_emb.device,
            )
        if markdown_embeddings is not None:
            markdown_emb = F.normalize(markdown_embeddings.to(anchor_emb.dtype), dim=-1)
            markdown_logits = torch.einsum("bd,bkd->bk", anchor_emb, markdown_emb) / self.temperature
            markdown_labels = markdown_labels.to(markdown_logits.device)
            logits = torch.cat([aug_logits, markdown_logits], dim=1)
            positive_mask = torch.zeros_like(logits, dtype=torch.bool)
            row_indices = torch.arange(anchor_emb.size(0), device=logits.device)
            positive_mask[row_indices, labels] = True
            positive_mask[row_indices, all_positive_emb.size(0) + markdown_labels] = True
            loss = (
                torch.logsumexp(logits, dim=1)
                - torch.logsumexp(logits.masked_fill(~positive_mask, float("-inf")), dim=1)
            ).mean()
            markdown_positive_emb = markdown_emb[:, 0, :]
            with torch.no_grad():
                recall1 = positive_mask.gather(1, logits.argsort(dim=1, descending=True))[:, 0].float().mean()

        return loss, {
            "anchor_embs": anchor_emb,
            "positive_embs": positive_emb,
            "aug_positive_embs": positive_emb,
            "markdown_positive_embs": markdown_positive_emb,
            "subject_ids": subject_ids,
            "recall@1": recall1.detach(),
        }


class ContrastiveDataset(Dataset):
    def __init__(self, base_datasets, is_eval: bool = False):
        self.base_datasets = base_datasets
        self.is_eval = is_eval
        self.index = []
        for dataset_idx, (_, dataset) in enumerate(self.base_datasets):
            for sample_idx in range(len(dataset)):
                self.index.append((dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        dataset_name, dataset = self.base_datasets[dataset_idx]
        sample = dataset[sample_idx]
        sample_info = dataset.sample_info[sample_idx]
        return {
            "table": sample.get("measurement_table"),
            "sample_key": build_sample_key(sample_info),
            "subject_id": build_subject_id(dataset_name, sample_info),
            "idx": idx,
            "is_eval": self.is_eval,
        }


def build_sample_key(sample_info):
    if "sample_id" in sample_info:
        return str(sample_info["sample_id"])
    return (
        f"{sample_info.get('subject_id', '')}|"
        f"{sample_info.get('context_begin', '')}|"
        f"{sample_info.get('context_end', '')}"
    )


def build_subject_id(dataset_name, sample_info):
    if dataset_name == "mimic_iv":
        return str(sample_info.get("subject_id", ""))
    return str(sample_info.get("patient_id", ""))


def normalize_markdown_embedding_keys(markdown_embeddings):
    normalized_embeddings = {}
    for key, value in markdown_embeddings.items():
        parts = str(key).split("|")
        if len(parts) >= 4 and parts[1] == "contrastive_learning":
            normalized_key = f"{parts[0]}|{parts[2]}|{parts[3]}"
        else:
            normalized_key = str(key)
        normalized_embeddings[normalized_key] = value
    return normalized_embeddings


def markdown_subject_id(key):
    parts = str(key).split("|")
    if parts[0] in {"eicu", "ehrshot"}:
        return parts[1]
    return parts[0]


def sample_low_overlap_views(
    table: pd.DataFrame,
    view_keep_ratio: float,
    max_view_overlap_ratio: float,
    min_table_rows: int,
    rng: random.Random,
):
    table = table.reset_index(drop=True)
    keep_count = int(round(len(table) * view_keep_ratio))
    keep_count = max(min_table_rows, min(len(table), keep_count))

    anchor_indices = sorted(rng.sample(range(len(table)), keep_count))
    anchor_set = set(anchor_indices)
    remaining_indices = [idx for idx in range(len(table)) if idx not in anchor_set]

    min_overlap_count = max(0, keep_count - len(remaining_indices))
    requested_overlap_count = int(round(keep_count * max_view_overlap_ratio))
    overlap_count = max(min_overlap_count, min(keep_count, requested_overlap_count))

    overlap_indices = rng.sample(anchor_indices, overlap_count) if overlap_count > 0 else []
    non_overlap_count = keep_count - overlap_count
    non_overlap_indices = rng.sample(remaining_indices, non_overlap_count) if non_overlap_count > 0 else []
    positive_indices = sorted(overlap_indices + non_overlap_indices)

    anchor_view = table.iloc[anchor_indices].reset_index(drop=True)
    positive_view = table.iloc[positive_indices].reset_index(drop=True)
    return anchor_view, positive_view


class ContrastiveDataCollator:
    def __init__(
        self,
        text_to_idx: dict[str, int],
        pad_idx: int,
        type_vocab: dict[str, int],
        max_table_len: Optional[int],
        min_table_rows: int,
        view_keep_ratio: float,
        max_view_overlap_ratio: float,
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
        self.view_keep_ratio = view_keep_ratio
        self.max_view_overlap_ratio = max_view_overlap_ratio
        self.markdown_key_to_idx = markdown_key_to_idx
        self.markdown_keys = markdown_keys
        self.markdown_subject_ids = markdown_subject_ids
        self.markdown_candidate_count = markdown_candidate_count
        self.augmentation_seed = augmentation_seed

    def sample_markdown_candidates(self, sample_key: str, subject_id: str, rng: random.Random):
        positive_idx = self.markdown_key_to_idx[sample_key]
        candidate_indices = [positive_idx]
        seen_indices = {positive_idx}
        target_negative_count = self.markdown_candidate_count - 1
        max_attempts = target_negative_count * 10 + 100

        attempts = 0
        while len(candidate_indices) < self.markdown_candidate_count and attempts < max_attempts:
            candidate_idx = rng.randrange(len(self.markdown_keys))
            attempts += 1
            if candidate_idx in seen_indices:
                continue
            if self.markdown_subject_ids[candidate_idx] == subject_id:
                continue
            candidate_indices.append(candidate_idx)
            seen_indices.add(candidate_idx)

        if len(candidate_indices) < self.markdown_candidate_count:
            for candidate_idx in range(len(self.markdown_keys)):
                if candidate_idx in seen_indices:
                    continue
                if self.markdown_subject_ids[candidate_idx] == subject_id:
                    continue
                candidate_indices.append(candidate_idx)
                seen_indices.add(candidate_idx)
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
        markdown_labels = []
        subject_ids = []

        for item in batch:
            table = item["table"]
            if table is None or table.empty:
                continue
            sample_key = item["sample_key"]
            if sample_key not in self.markdown_key_to_idx:
                continue
            if self.max_table_len is not None:
                table = table.tail(self.max_table_len).reset_index(drop=True)
            if len(table) < self.min_table_rows:
                continue

            subject_ids.append(item["subject_id"])
            if item["is_eval"]:
                rng = random.Random(self.augmentation_seed + int(item["idx"]))
            else:
                rng = random
            anchor_view, positive_view = sample_low_overlap_views(
                table,
                view_keep_ratio=self.view_keep_ratio,
                max_view_overlap_ratio=self.max_view_overlap_ratio,
                min_table_rows=self.min_table_rows,
                rng=rng,
            )
            anchor_tables.append(anchor_view)
            markdown_candidate_indices.append(self.sample_markdown_candidates(sample_key, item["subject_id"], rng))
            markdown_labels.append(0)
            positive_tables.append(positive_view)

        if len(anchor_tables) == 0:
            raise ValueError("All samples in this batch have fewer than two table rows.")

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

        model_inputs = {
            **anchor_tensors,
            "positive_": positive_tensors,
            "markdown_candidate_indices": torch.stack(markdown_candidate_indices),
            "markdown_labels": torch.tensor(markdown_labels, dtype=torch.long),
            "subject_ids": subject_ids,
            "labels": torch.zeros(len(anchor_tables), dtype=torch.long),
        }
        return model_inputs


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
        all_anchor_embs = []
        all_aug_positive_embs = []
        all_markdown_positive_embs = []
        all_subject_ids = []
        total_loss = 0.0
        num_batches = 0

        for inputs in eval_dataloader:
            inputs = self._prepare_inputs(inputs)
            subject_ids = inputs.get("subject_ids")
            with torch.no_grad():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            total_loss += loss.mean().item()
            num_batches += 1
            all_anchor_embs.append(outputs["anchor_embs"].detach().float().cpu())
            all_aug_positive_embs.append(outputs["aug_positive_embs"].detach().float().cpu())
            if outputs["markdown_positive_embs"] is not None:
                all_markdown_positive_embs.append(outputs["markdown_positive_embs"].detach().float().cpu())
            all_subject_ids.extend(subject_ids)

        model.train()

        local_anchor = torch.cat(all_anchor_embs, dim=0)
        local_aug_positive = torch.cat(all_aug_positive_embs, dim=0)
        local_markdown_positive = (
            torch.cat(all_markdown_positive_embs, dim=0)
            if len(all_markdown_positive_embs) > 0
            else None
        )
        local_subject_ids = all_subject_ids
        rank = 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            anchor_list = [None] * world_size
            aug_positive_list = [None] * world_size
            markdown_positive_list = [None] * world_size
            subject_id_list = [None] * world_size
            dist.all_gather_object(anchor_list, local_anchor)
            dist.all_gather_object(aug_positive_list, local_aug_positive)
            dist.all_gather_object(markdown_positive_list, local_markdown_positive)
            dist.all_gather_object(subject_id_list, local_subject_ids)
            if rank == 0:
                anchor_all = torch.cat(anchor_list, dim=0)
                aug_positive_all = torch.cat(aug_positive_list, dim=0)
                markdown_positive_all = (
                    torch.cat(markdown_positive_list, dim=0)
                    if markdown_positive_list[0] is not None
                    else None
                )
                subject_ids_all = [subject_id for rank_subject_ids in subject_id_list for subject_id in rank_subject_ids]
            else:
                anchor_all = local_anchor
                aug_positive_all = local_aug_positive
                markdown_positive_all = local_markdown_positive
                subject_ids_all = local_subject_ids
        else:
            anchor_all = local_anchor
            aug_positive_all = local_aug_positive
            markdown_positive_all = local_markdown_positive
            subject_ids_all = local_subject_ids

        loss_tensor = torch.tensor(total_loss / max(num_batches, 1))
        if dist.is_available() and dist.is_initialized():
            loss_tensor = loss_tensor.to(self.args.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            loss_tensor = loss_tensor.cpu()

        metrics = {f"{metric_key_prefix}_loss": loss_tensor.item()}

        if rank == 0:
            if markdown_positive_all is not None:
                positive_all = torch.cat([aug_positive_all, markdown_positive_all], dim=0)
                positive_subject_ids = subject_ids_all + subject_ids_all
                positive_indices = [
                    [row_idx, row_idx + anchor_all.shape[0]]
                    for row_idx in range(anchor_all.shape[0])
                ]
            else:
                positive_all = aug_positive_all
                positive_subject_ids = subject_ids_all
                positive_indices = [[row_idx] for row_idx in range(anchor_all.shape[0])]
            recall_metrics = compute_recall_metrics(
                anchor_all,
                positive_all,
                query_subject_ids=subject_ids_all,
                positive_subject_ids=positive_subject_ids,
                positive_indices=positive_indices,
            )
            metrics.update({f"{metric_key_prefix}_{k}": v for k, v in recall_metrics.items()})
            print(f"[Eval] step={self.state.global_step} N={anchor_all.shape[0]} " + " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            ))

        if dist.is_available() and dist.is_initialized():
            metrics_list = [metrics] if rank == 0 else [None]
            dist.broadcast_object_list(metrics_list, src=0)
            metrics = metrics_list[0]

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
):
    num_queries = anchor_embs.size(0)
    num_positives = positive_embs.size(0)
    if positive_indices is None:
        positive_indices = [[row_idx] for row_idx in range(num_queries)]

    recall_hits = {k: 0.0 for k in [1, 5, 10, 50] if k <= num_positives}
    max_k = max(recall_hits.keys(), default=1)
    reciprocal_rank_sum = 0.0

    positive_embs_t = positive_embs.t()
    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        sim = torch.matmul(anchor_embs[start:end], positive_embs_t)

        if query_subject_ids is not None and positive_subject_ids is not None:
            mask_rows = []
            for local_row_idx, subject_id in enumerate(query_subject_ids[start:end]):
                global_row_idx = start + local_row_idx
                positive_index_set = set(positive_indices[global_row_idx])
                mask_rows.append(
                    [
                        subject_id == other_subject_id and col_idx not in positive_index_set
                        for col_idx, other_subject_id in enumerate(positive_subject_ids)
                    ]
                )
            same_subject_mask = torch.tensor(mask_rows, dtype=torch.bool, device=sim.device)
            sim = sim.masked_fill(same_subject_mask, float("-inf"))

        topk_indices = torch.topk(sim, k=max_k, dim=1).indices
        for local_row_idx in range(end - start):
            global_row_idx = start + local_row_idx
            positive_index_set = set(positive_indices[global_row_idx])
            topk_row = topk_indices[local_row_idx].tolist()
            for k in recall_hits:
                if any(idx in positive_index_set for idx in topk_row[:k]):
                    recall_hits[k] += 1.0

            pos_idx = torch.tensor(positive_indices[global_row_idx], dtype=torch.long, device=sim.device)
            best_positive_score = sim[local_row_idx, pos_idx].max()
            rank = (sim[local_row_idx] > best_positive_score).sum().item() + 1
            reciprocal_rank_sum += 1.0 / rank

    metrics = {}
    for k, hits in recall_hits.items():
        metrics[f"recall@{k}"] = hits / max(num_queries, 1)
    metrics["mrr"] = reciprocal_rank_sum / max(num_queries, 1)
    return metrics


def filter_samples(samples, min_table_rows: int, max_samples: Optional[int], markdown_key_to_idx):
    filtered_samples = []
    for sample in samples:
        table_length = sample.get("table_length")
        has_table_rows = pd.isna(table_length) or int(table_length) >= min_table_rows
        has_markdown_embedding = build_sample_key(sample) in markdown_key_to_idx
        if has_table_rows and has_markdown_embedding:
            filtered_samples.append(sample)
    samples = filtered_samples
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def load_table_embeddings(cache_paths: List[str]):
    embedding_cache = {}
    for cache_path in cache_paths:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        embedding_cache.update(data["embeddings"])
        text_dim = int(data["text_dim"])
        print(f"Loaded {len(data['embeddings'])} embeddings from {cache_path}")

    vocab_keys = list(embedding_cache.keys())
    text_to_idx = build_text_to_idx(vocab_keys)
    matrix = torch.empty(len(vocab_keys), text_dim)
    for idx, text in enumerate(vocab_keys):
        matrix[idx] = embedding_cache[text]
    return text_dim, vocab_keys, text_to_idx, matrix


def load_markdown_embeddings(paths: List[str]):
    markdown_embeddings = {}
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=True)
        markdown_embeddings.update(normalize_markdown_embedding_keys(data))
        print(f"Loaded {len(data)} markdown embeddings from {path}")

    markdown_keys = list(markdown_embeddings.keys())
    if len(markdown_keys) == 0:
        raise ValueError("No markdown embeddings loaded.")
    first_embedding = markdown_embeddings[markdown_keys[0]]
    markdown_embedding_matrix = torch.empty(
        len(markdown_keys),
        first_embedding.numel(),
        dtype=first_embedding.dtype,
    )
    for idx, key in enumerate(markdown_keys):
        markdown_embedding_matrix[idx] = markdown_embeddings[key].reshape(-1)

    markdown_key_to_idx = {key: idx for idx, key in enumerate(markdown_keys)}
    markdown_subject_ids = [markdown_subject_id(key) for key in markdown_keys]
    del markdown_embeddings
    return markdown_key_to_idx, markdown_keys, markdown_subject_ids, markdown_embedding_matrix


def get_embedding_cache_paths(data_args: DataArguments):
    cache_paths = []
    for dataset_name in data_args.dataset:
        if dataset_name == "mimic_iv":
            cache_paths.extend(data_args.table_text_embedding)
        elif dataset_name == "eicu":
            cache_paths.extend(data_args.eicu_table_text_embedding)
        elif dataset_name == "ehrshot":
            cache_paths.extend(data_args.ehrshot_table_text_embedding)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return cache_paths


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
        return (
            dataset_name,
            MIMICIV(
                root_dir=data_args.root_dir,
                sample_info_path=sample_info_path,
                lazy_mode=True,
                shuffle=False,
                table_mode="table_only",
                max_samples=None,
                use_table_length_cache=False,
            ),
        )
    if dataset_name == "eicu":
        return (
            dataset_name,
            EICUDataset(
                root_dir=data_args.eicu_root_dir,
                processed_dir=data_args.eicu_processed_dir,
                sample_info_path=sample_info_path,
                task_name=None,
                lazy_mode=True,
                shuffle=False,
                table_mode="table_only",
            ),
        )
    if dataset_name == "ehrshot":
        return (
            dataset_name,
            EHRSHOTDataset(
                root_dir=data_args.ehrshot_root_dir,
                sample_info_path=sample_info_path,
                task_name=None,
                lazy_mode=True,
                table_mode="table_only",
            ),
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_base_datasets(data_args: DataArguments, split: str, markdown_key_to_idx):
    datasets = []
    for dataset_name in data_args.dataset:
        if dataset_name == "mimic_iv":
            sample_info_path = data_args.sample_info_path if split == "train" else data_args.val_sample_info_path
            max_samples = data_args.max_train_samples if split == "train" else data_args.max_eval_samples
        elif dataset_name == "eicu":
            sample_info_path = data_args.eicu_sample_info_path if split == "train" else data_args.eicu_val_sample_info_path
            max_samples = data_args.max_train_samples if split == "train" else data_args.max_eval_samples
        elif dataset_name == "ehrshot":
            sample_info_path = data_args.ehrshot_sample_info_path if split == "train" else data_args.ehrshot_val_sample_info_path
            max_samples = data_args.max_train_samples if split == "train" else data_args.max_eval_samples
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        dataset_name, dataset = build_one_dataset(dataset_name, data_args, sample_info_path)
        dataset.sample_info = filter_samples(
            dataset.sample_info,
            data_args.min_table_rows,
            max_samples,
            markdown_key_to_idx,
        )
        print(f"{split} {dataset_name} samples: {len(dataset.sample_info)}")
        datasets.append((dataset_name, dataset))
    return datasets


def main():
    parser = HfArgumentParser((DataArguments, TrainingArgumentsCustom))
    data_args, training_args = parser.parse_args_into_dataclasses()

    print("Stage 2: Contrastive Learning - Table <-> Augmented Table/Markdown")
    print(f"Datasets: {', '.join(data_args.dataset)}")
    print(f"Table text embeddings: {get_embedding_cache_paths(data_args)}")
    print(f"Markdown embeddings: {get_markdown_embedding_paths(data_args)}")
    print(f"Markdown candidate count: {data_args.markdown_candidate_count}")
    print(f"Pretrained path: {data_args.pretrained_path}")
    print(f"View keep ratio: {data_args.view_keep_ratio}")
    print(f"Max view overlap ratio: {data_args.max_view_overlap_ratio}")

    text_dim, vocab_keys, text_to_idx, embedding_matrix = load_table_embeddings(get_embedding_cache_paths(data_args))
    markdown_key_to_idx, markdown_keys, markdown_subject_ids, markdown_embedding_matrix = load_markdown_embeddings(
        get_markdown_embedding_paths(data_args)
    )

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
    )
    model = load_model_weights(model, data_args.pretrained_path)

    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    train_bases = build_base_datasets(data_args, "train", markdown_key_to_idx)
    eval_bases = build_base_datasets(data_args, "eval", markdown_key_to_idx)
    train_size = sum(len(dataset) for _, dataset in train_bases)
    eval_size = sum(len(dataset) for _, dataset in eval_bases)
    if train_size == 0:
        raise ValueError("No training samples left after table-length and markdown-embedding filtering.")
    if eval_size == 0:
        raise ValueError("No eval samples left after table-length and markdown-embedding filtering.")

    train_dataset = ContrastiveDataset(train_bases, is_eval=False)
    eval_dataset = ContrastiveDataset(eval_bases, is_eval=True)

    collator = ContrastiveDataCollator(
        text_to_idx=text_to_idx,
        pad_idx=0,
        type_vocab=type_vocab,
        max_table_len=data_args.max_table_len,
        min_table_rows=data_args.min_table_rows,
        view_keep_ratio=data_args.view_keep_ratio,
        max_view_overlap_ratio=data_args.max_view_overlap_ratio,
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
