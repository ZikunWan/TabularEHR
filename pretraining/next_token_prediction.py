import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.mimic.mimic_dataset import MIMICIV
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.next_token_decoder import NextTokenPredictionModel
from utils.collate import build_table_token_tensors
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)


LOSS_COMPONENT_NAMES = (
    "category_loss",
    "item_loss",
    "unit_loss",
    "value_loss",
    "weighted_category_loss",
    "weighted_item_loss",
    "weighted_unit_loss",
    "weighted_value_loss",
)


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in [-1, 0]:
        print(*args, **kwargs)


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    train_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/next_token_prediction.csv"
    )
    table_text_embedding: str = field(
        default="/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings.pt"
    )
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    max_train_samples: Optional[int] = field(default=None)
    max_table_len: Optional[int] = field(default=32768)
    min_table_rows: int = field(default=2)


@dataclass
class NextTokenTrainingArguments(TrainingArguments):
    output_dir: str = field(default="/data/zikun_workspace/checkpoints/pretraining/next_token_prediction")
    num_train_epochs: float = field(default=1)
    per_device_train_batch_size: int = field(default=8)
    learning_rate: float = field(default=1e-4)
    warmup_steps: int = field(default=100)
    weight_decay: float = field(default=0.01)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=100)
    save_total_limit: int = field(default=1)
    bf16: bool = field(default=True)
    dataloader_num_workers: int = field(default=32)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="wandb")
    wandb_project: str = field(default="Next_Token_Prediction")


class NextTokenDataCollator:
    def __init__(
        self,
        text_to_idx: dict[str, int],
        pad_idx: int,
        type_vocab: dict[str, int],
        max_table_len: Optional[int],
        min_table_rows: int,
    ):
        self.text_to_idx = text_to_idx
        self.pad_idx = pad_idx
        self.type_vocab = type_vocab
        self.max_table_len = max_table_len
        self.min_table_rows = min_table_rows

    def __call__(self, batch):
        tables = []
        for sample in batch:
            table = sample.get("measurement_table")
            if table is None or table.empty:
                continue
            if self.max_table_len is not None:
                table = table.tail(self.max_table_len).reset_index(drop=True)
            if len(table) < self.min_table_rows:
                continue
            tables.append(table)

        if len(tables) == 0:
            raise ValueError("All samples in this batch have fewer than two table rows.")

        return build_table_token_tensors(
            tables,
            text_to_idx=self.text_to_idx,
            pad_idx=self.pad_idx,
            type_vocab=self.type_vocab,
        )


def load_type_vocab(path: str) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_by_table_rows(samples, min_table_rows: int):
    kept = []
    for sample in samples:
        table_length = sample.get("table_length")
        if pd.notna(table_length) and int(table_length) < min_table_rows:
            continue
        kept.append(sample)
    return kept


def build_dataset(root_dir: str, sample_info_path: str, max_samples: Optional[int], min_table_rows: int, shuffle: bool):
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    dataset = MIMICIV(
        root_dir=root_dir,
        sample_info_path=sample_info_path,
        lazy_mode=True,
        shuffle=shuffle,
        table_mode="table_only",
        max_samples=None,
        use_table_length_cache=False,
    )
    dataset.sample_info = filter_by_table_rows(dataset.sample_info, min_table_rows)
    if max_samples is not None:
        dataset.sample_info = dataset.sample_info[:max_samples]
    if len(dataset.sample_info) == 0:
        raise ValueError(f"No samples left after min_table_rows={min_table_rows} filtering: {sample_info_path}")
    return dataset


class NextTokenPredictionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if not hasattr(self, "_loss_component_sums"):
            self._loss_component_sums = {name: 0.0 for name in LOSS_COMPONENT_NAMES}
            self._loss_component_count = 0

        for name in LOSS_COMPONENT_NAMES:
            value = getattr(outputs, name, None)
            if value is not None:
                self._loss_component_sums[name] += value.detach().float().item()
        self._loss_component_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if hasattr(self, "_loss_component_sums") and self._loss_component_count > 0:
            logs = dict(logs)
            for name, value in self._loss_component_sums.items():
                logs[name] = value / self._loss_component_count
            self._loss_component_sums = {name: 0.0 for name in LOSS_COMPONENT_NAMES}
            self._loss_component_count = 0
        super().log(logs, start_time=start_time)


def main():
    parser = HfArgumentParser((DataArguments, NextTokenTrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    set_seed(training_args.seed)

    embedding_cache, text_dim = load_embedding_cache(data_args.table_text_embedding)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    special_indices = get_special_token_indices(text_to_idx)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)

    type_vocab = load_type_vocab(data_args.type_vocab_file)
    type_vocab_size = max(type_vocab.values()) + 1

    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=type_vocab_size,
        max_table_len=data_args.max_table_len,
    )
    model = NextTokenPredictionModel(
        config=config,
        embedding_matrix=embedding_matrix,
    )

    train_dataset = build_dataset(
        data_args.root_dir,
        data_args.train_info_path,
        data_args.max_train_samples,
        data_args.min_table_rows,
        shuffle=True,
    )

    collator = NextTokenDataCollator(
        text_to_idx=text_to_idx,
        pad_idx=special_indices["pad_idx"],
        type_vocab=type_vocab,
        max_table_len=config.max_table_len,
        min_table_rows=data_args.min_table_rows,
    )

    rank0_print(f"Train samples: {len(train_dataset)}")
    rank0_print(f"Table vocab size: {len(vocab_keys)}, text_dim={text_dim}, type_vocab_size={type_vocab_size}")

    trainer = NextTokenPredictionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
