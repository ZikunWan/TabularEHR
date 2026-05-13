import builtins
import json
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

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

from dataset.mimic.mimic_dataset import MIMICIV
from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D
from utils.collate import build_table_token_tensors
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
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
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/next_token_prediction.csv"
    )
    val_sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val/next_token_prediction.csv"
    )
    table_text_embedding: str = field(
        default="/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings.pt"
    )
    markdown_embedding_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/embeddings/table_free_text/embeddings.pt"
    )
    markdown_candidate_count: int = field(default=1024)
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    pretrained_path: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_table_len: Optional[int] = field(default=16384)
    min_table_rows: int = field(default=2)
    table_drop_ratio: float = field(default=0.15)
    time_window_ratio: float = field(default=0.85)
    value_mask_ratio: float = field(default=0.10)
    unit_mask_ratio: float = field(default=0.10)
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
        temperature: float = 0.07,
    ):
        super().__init__(config)
        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.temperature = temperature
        self.text_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        pool_hidden_size = config.dim_out if config.dim_out is not None else config.dim
        self.table_pooling = AttentionPooling(pool_hidden_size)

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
        item_emb = self.text_embedding(item_ids)
        unit_emb = self.text_embedding(unit_ids)
        value_emb = self.text_embedding(value_text_ids)
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

    def forward(self, positive_, markdown_embeddings=None, markdown_labels=None, subject_ids=None, **anchor_inputs):
        anchor_emb = F.normalize(self.encode_table(**anchor_inputs), dim=-1)
        positive_emb = F.normalize(self.encode_table(**positive_), dim=-1)

        all_positive_emb, label_offset = all_gather_with_grad(positive_emb)
        aug_logits = torch.matmul(anchor_emb, all_positive_emb.t()) / self.temperature
        labels = label_offset + torch.arange(anchor_emb.size(0), device=aug_logits.device)
        loss = F.cross_entropy(aug_logits, labels)

        with torch.no_grad():
            recall1 = (aug_logits.argmax(dim=1) == labels).float().mean()

        markdown_positive_emb = None
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
    def __init__(self, base_dataset, is_eval: bool = False):
        self.base_dataset = base_dataset
        self.is_eval = is_eval

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample_info = self.base_dataset.sample_info[idx]
        return {
            "table": sample.get("measurement_table"),
            "sample_key": build_sample_key(sample_info),
            "subject_id": str(sample_info.get("subject_id", "")),
            "idx": idx,
            "is_eval": self.is_eval,
        }


def build_sample_key(sample_info):
    return (
        f"{sample_info.get('subject_id', '')}|"
        f"{sample_info.get('context_begin', '')}|"
        f"{sample_info.get('context_end', '')}"
    )


def normalize_markdown_embedding_keys(markdown_embeddings):
    normalized_embeddings = {}
    for key, value in markdown_embeddings.items():
        parts = str(key).split("|")
        normalized_key = f"{parts[0]}|{parts[2]}|{parts[3]}"
        normalized_embeddings[normalized_key] = value
    return normalized_embeddings


def augment_table(
    table: pd.DataFrame,
    drop_ratio: float,
    time_window_ratio: float,
    value_mask_ratio: float,
    unit_mask_ratio: float,
    min_table_rows: int,
    rng: random.Random,
):
    table = table.reset_index(drop=True)
    if len(table) <= min_table_rows:
        return table

    augmentation_type = rng.choice(["time_window", "row_drop", "value_mask", "unit_mask", "time_shuffle"])

    if augmentation_type == "time_window" and time_window_ratio < 1.0:
        keep_count = int(round(len(table) * time_window_ratio))
        keep_count = max(min_table_rows, min(len(table), keep_count))
        start_idx = rng.randrange(0, len(table) - keep_count + 1)
        return table.iloc[start_idx:start_idx + keep_count].reset_index(drop=True)

    if augmentation_type == "row_drop":
        keep_count = int(round(len(table) * (1.0 - drop_ratio)))
        keep_count = max(min_table_rows, min(len(table), keep_count))
        keep_indices = sorted(rng.sample(range(len(table)), keep_count))
        return table.iloc[keep_indices].reset_index(drop=True)

    if augmentation_type == "value_mask" and value_mask_ratio > 0 and "Value" in table.columns:
        mask_count = int(round(len(table) * value_mask_ratio))
        mask_count = min(len(table), mask_count)
        if mask_count > 0:
            mask_indices = rng.sample(range(len(table)), mask_count)
            table.loc[mask_indices, "Value"] = pd.NA
        return table

    if augmentation_type == "unit_mask" and unit_mask_ratio > 0 and "Unit" in table.columns:
        mask_count = int(round(len(table) * unit_mask_ratio))
        mask_count = min(len(table), mask_count)
        if mask_count > 0:
            mask_indices = rng.sample(range(len(table)), mask_count)
            table.loc[mask_indices, "Unit"] = pd.NA
        return table

    if augmentation_type == "time_shuffle" and "Time" in table.columns:
        shuffled_indices = []
        for _, group in table.groupby("Time", sort=False):
            group_indices = group.index.tolist()
            rng.shuffle(group_indices)
            shuffled_indices.extend(group_indices)
        return table.loc[shuffled_indices].reset_index(drop=True)

    return table


class ContrastiveDataCollator:
    def __init__(
        self,
        text_to_idx: dict[str, int],
        pad_idx: int,
        type_vocab: dict[str, int],
        max_table_len: Optional[int],
        min_table_rows: int,
        table_drop_ratio: float,
        time_window_ratio: float,
        value_mask_ratio: float,
        unit_mask_ratio: float,
        markdown_embeddings: dict[str, torch.Tensor],
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
        self.table_drop_ratio = table_drop_ratio
        self.time_window_ratio = time_window_ratio
        self.value_mask_ratio = value_mask_ratio
        self.unit_mask_ratio = unit_mask_ratio
        self.markdown_embeddings = markdown_embeddings
        self.markdown_keys = list(markdown_embeddings.keys())
        self.markdown_subject_ids = {key: str(key).split("|", 1)[0] for key in self.markdown_keys}
        self.markdown_candidate_count = markdown_candidate_count
        self.augmentation_seed = augmentation_seed

    def sample_markdown_candidates(self, sample_key: str, subject_id: str, rng: random.Random):
        candidate_keys = [sample_key]
        seen_keys = {sample_key}
        target_negative_count = self.markdown_candidate_count - 1
        max_attempts = target_negative_count * 10 + 100

        attempts = 0
        while len(candidate_keys) < self.markdown_candidate_count and attempts < max_attempts:
            key = self.markdown_keys[rng.randrange(len(self.markdown_keys))]
            attempts += 1
            if key in seen_keys:
                continue
            if self.markdown_subject_ids[key] == subject_id:
                continue
            candidate_keys.append(key)
            seen_keys.add(key)

        if len(candidate_keys) < self.markdown_candidate_count:
            for key in self.markdown_keys:
                if key in seen_keys:
                    continue
                if self.markdown_subject_ids[key] == subject_id:
                    continue
                candidate_keys.append(key)
                seen_keys.add(key)
                if len(candidate_keys) == self.markdown_candidate_count:
                    break

        if len(candidate_keys) < self.markdown_candidate_count:
            raise ValueError(
                f"Only found {len(candidate_keys)} markdown candidates for subject_id={subject_id}; "
                f"required {self.markdown_candidate_count}."
            )

        return torch.stack([self.markdown_embeddings[key] for key in candidate_keys])

    def __call__(self, batch):
        anchor_tables = []
        positive_tables = []
        markdown_embeddings = []
        markdown_labels = []
        subject_ids = []

        for item in batch:
            table = item["table"]
            if table is None or table.empty:
                continue
            sample_key = item["sample_key"]
            if sample_key not in self.markdown_embeddings:
                continue
            if self.max_table_len is not None:
                table = table.tail(self.max_table_len).reset_index(drop=True)
            if len(table) < self.min_table_rows:
                continue

            anchor_tables.append(table)
            subject_ids.append(item["subject_id"])
            if item["is_eval"]:
                rng = random.Random(self.augmentation_seed + int(item["idx"]))
            else:
                rng = random
            markdown_embeddings.append(self.sample_markdown_candidates(sample_key, item["subject_id"], rng))
            markdown_labels.append(0)
            positive_tables.append(
                augment_table(
                    table,
                    drop_ratio=self.table_drop_ratio,
                    time_window_ratio=self.time_window_ratio,
                    value_mask_ratio=self.value_mask_ratio,
                    unit_mask_ratio=self.unit_mask_ratio,
                    min_table_rows=self.min_table_rows,
                    rng=rng,
                )
            )

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
            "markdown_embeddings": torch.stack(markdown_embeddings),
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
):
    sim = torch.matmul(anchor_embs, positive_embs.t())
    if positive_indices is None:
        positive_indices = [[row_idx] for row_idx in range(sim.size(0))]
    if query_subject_ids is not None and positive_subject_ids is not None:
        positive_index_sets = [set(indices) for indices in positive_indices]
        same_subject_mask = torch.tensor(
            [
                [
                    subject_id == other_subject_id and col_idx not in positive_index_sets[row_idx]
                    for col_idx, other_subject_id in enumerate(positive_subject_ids)
                ]
                for row_idx, subject_id in enumerate(query_subject_ids)
            ],
            dtype=torch.bool,
            device=sim.device,
        )
        sim = sim.masked_fill(same_subject_mask, float("-inf"))
    sorted_indices = torch.argsort(sim, dim=1, descending=True)
    positive_mask = torch.zeros_like(sim, dtype=torch.bool)
    for row_idx, indices in enumerate(positive_indices):
        positive_mask[row_idx, indices] = True
    matches = positive_mask.gather(1, sorted_indices)

    metrics = {}
    for k in [1, 5, 10, 50]:
        if k <= sim.size(0):
            metrics[f"recall@{k}"] = matches[:, :k].any(dim=1).float().mean().item()
    ranks = matches.float().argmax(dim=1) + 1
    metrics["mrr"] = (1.0 / ranks.float()).mean().item()
    return metrics


def filter_samples(samples, min_table_rows: int, max_samples: Optional[int], markdown_embeddings):
    filtered_samples = []
    for sample in samples:
        table_length = sample.get("table_length")
        has_table_rows = pd.isna(table_length) or int(table_length) >= min_table_rows
        has_markdown_embedding = build_sample_key(sample) in markdown_embeddings
        if has_table_rows and has_markdown_embedding:
            filtered_samples.append(sample)
    samples = filtered_samples
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def main():
    parser = HfArgumentParser((DataArguments, TrainingArgumentsCustom))
    data_args, training_args = parser.parse_args_into_dataclasses()

    print("Stage 2: Contrastive Learning - Table <-> Augmented Table/Markdown")
    print(f"Train sample info: {data_args.sample_info_path}")
    print(f"Val sample info: {data_args.val_sample_info_path}")
    print(f"Table text embeddings: {data_args.table_text_embedding}")
    print(f"Markdown embeddings: {data_args.markdown_embedding_path}")
    print(f"Markdown candidate count: {data_args.markdown_candidate_count}")
    print(f"Pretrained path: {data_args.pretrained_path}")
    print(f"Drop ratio: {data_args.table_drop_ratio}")
    print(f"Time window ratio: {data_args.time_window_ratio}")
    print(f"Value mask ratio: {data_args.value_mask_ratio}")
    print(f"Unit mask ratio: {data_args.unit_mask_ratio}")

    embedding_cache, text_dim = load_embedding_cache(data_args.table_text_embedding)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    special_indices = get_special_token_indices(text_to_idx)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    markdown_embeddings = torch.load(data_args.markdown_embedding_path, map_location="cpu", weights_only=True)
    markdown_embeddings = normalize_markdown_embedding_keys(markdown_embeddings)

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
        temperature=data_args.temperature,
    )
    model = load_model_weights(model, data_args.pretrained_path)

    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    train_base = MIMICIV(
        root_dir=data_args.root_dir,
        sample_info_path=data_args.sample_info_path,
        lazy_mode=True,
        shuffle=False,
        table_mode="table_only",
        max_samples=None,
        use_table_length_cache=False,
    )
    eval_base = MIMICIV(
        root_dir=data_args.root_dir,
        sample_info_path=data_args.val_sample_info_path,
        lazy_mode=True,
        shuffle=False,
        table_mode="table_only",
        max_samples=None,
        use_table_length_cache=False,
    )

    train_samples = filter_samples(
        train_base.sample_info,
        data_args.min_table_rows,
        data_args.max_train_samples,
        markdown_embeddings,
    )
    eval_samples = filter_samples(
        eval_base.sample_info,
        data_args.min_table_rows,
        data_args.max_eval_samples,
        markdown_embeddings,
    )
    if len(train_samples) == 0:
        raise ValueError("No training samples left after table-length and markdown-embedding filtering.")
    if len(eval_samples) == 0:
        raise ValueError("No eval samples left after table-length and markdown-embedding filtering.")

    train_base.sample_info = train_samples
    eval_base.sample_info = eval_samples

    train_dataset = ContrastiveDataset(train_base, is_eval=False)
    eval_dataset = ContrastiveDataset(eval_base, is_eval=True)

    collator = ContrastiveDataCollator(
        text_to_idx=text_to_idx,
        pad_idx=special_indices["pad_idx"],
        type_vocab=type_vocab,
        max_table_len=data_args.max_table_len,
        min_table_rows=data_args.min_table_rows,
        table_drop_ratio=data_args.table_drop_ratio,
        time_window_ratio=data_args.time_window_ratio,
        value_mask_ratio=data_args.value_mask_ratio,
        unit_mask_ratio=data_args.unit_mask_ratio,
        markdown_embeddings=markdown_embeddings,
        markdown_candidate_count=data_args.markdown_candidate_count,
        augmentation_seed=data_args.augmentation_seed,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    print(f"Table vocab size: {len(vocab_keys)}, text_dim={text_dim}, type_vocab_size={type_vocab_size}")

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
