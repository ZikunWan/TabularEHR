import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EarlyStoppingCallback, HfArgumentParser, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.query_classifier import TaskQueryClassificationModel
from models.TableEncoder.text_encoder import TextEncoder
from utils.collate import build_table_token_tensors
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.metrics import compute_classification_metrics
from utils.weight_loader import load_model_weights


RISK_PREDICTION_TASKS = [
    "ED_Hospitalization",
    "ED_Inpatient_Mortality",
    "ED_ICU_Tranfer_12hour",
    "ED_Reattendance_3day",
    "ED_Critical_Outcomes",
    "Readmission_30day",
    "Readmission_60day",
    "Inpatient_Mortality",
    "LengthOfStay_3day",
    "LengthOfStay_7day",
    "ICU_Mortality_1day",
    "ICU_Mortality_2day",
    "ICU_Mortality_3day",
    "ICU_Mortality_7day",
    "ICU_Mortality_14day",
    "ICU_Stay_7day",
    "ICU_Stay_14day",
    "ICU_Readmission",
]


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in [-1, 0]:
        print(*args, **kwargs)


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    train_sample_info_path: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train")
    val_sample_info_path: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val")
    table_text_embedding: str = field(default="/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings.pt")
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    pretrained_path: Optional[str] = field(default="/data/zikun_workspace/checkpoints/pretraining/contrastive_learning")
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/mimic_iv/task_query_embeddings.pt")
    query_text_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/text_encoder_stage2/epoch_5.pt")
    query_text_encoder_base_model: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=128)
    max_table_len: Optional[int] = field(default=16384)
    min_table_rows: int = field(default=2)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    output_dir: str = field(default="/data/zikun_workspace/checkpoints/pretraining/task_query_classification")
    num_train_epochs: int = field(default=10)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
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
    wandb_project: Optional[str] = field(default="Task_Query_Classification")
    metric_for_best_model: str = field(default="eval_auroc")
    greater_is_better: bool = field(default=True)
    early_stopping_patience: int = field(default=10)

    def __post_init__(self):
        super().__post_init__()
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        if self.eval_strategy == "no":
            self.eval_strategy = "steps"
        self.load_best_model_at_end = True
        self.greater_is_better = True


def resolve_sample_info_paths(path_arg: str):
    paths = []
    for raw_path in path_arg.split(","):
        path = raw_path.strip()
        if not path:
            continue
        if os.path.isdir(path):
            for task_name in RISK_PREDICTION_TASKS:
                csv_path = os.path.join(path, f"{task_name}.csv")
                if os.path.exists(csv_path):
                    paths.append(csv_path)
        else:
            paths.append(path)
    return paths


def load_checkpoint_state_dict(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def load_query_text_encoder(model_path: str, base_model_path: str, device: torch.device):
    model_name = base_model_path if model_path.endswith(".pt") else model_path
    model = TextEncoder(model_name)
    if model_path.endswith(".pt"):
        state_dict = load_checkpoint_state_dict(model_path)
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def local_rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def wait_for_query_cache(cache_path: str, task_names):
    while True:
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_embeddings = cache["embeddings"]
            if all(task_name in cached_embeddings for task_name in task_names):
                return cache
        time.sleep(2)


def build_query_embeddings(
    task_names,
    cache_path: str,
    model_path: str,
    base_model_path: str,
    max_length: int,
):
    task_names = sorted(set(task_names))
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_embeddings = cache["embeddings"]
        if all(task_name in cached_embeddings for task_name in task_names):
            return cached_embeddings, int(cache["text_dim"])

    if not local_rank0():
        cache = wait_for_query_cache(cache_path, task_names)
        return cache["embeddings"], int(cache["text_dim"])

    task_info = get_task_info()
    query_texts = [task_info[task_name]["instruction"] for task_name in task_names]

    tokenizer_path = base_model_path if model_path.endswith(".pt") else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_query_text_encoder(model_path, base_model_path, device)
    model.eval()

    tokens = tokenizer(
        query_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        query_embeds = model.encode_text(tokens).cpu()

    embeddings = {
        task_name: query_embeds[idx]
        for idx, task_name in enumerate(task_names)
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": model.hidden_size,
            "model_path": model_path,
            "base_model_path": base_model_path,
        },
        cache_path,
    )
    rank0_print(f"Saved query embeddings to {cache_path}")
    return embeddings, int(model.hidden_size)


def parse_binary_label(value) -> int:
    label = str(value).strip().strip('"').strip("'").strip().lower()
    if label == "yes":
        return 1
    if label == "no":
        return 0
    return int(float(label))


class TaskQueryDataset(Dataset):
    def __init__(self, root_dir: str, sample_info_paths, max_samples: Optional[int]):
        self.datasets = [
            MIMICIV(
                root_dir=root_dir,
                sample_info_path=sample_info_path,
                lazy_mode=True,
                shuffle=False,
                table_mode="table_only",
                max_samples=None,
                use_table_length_cache=False,
            )
            for sample_info_path in sample_info_paths
        ]
        self.index = []
        if max_samples is not None:
            positions = [0] * len(self.datasets)
            while len(self.index) < max_samples:
                added = False
                for dataset_idx, dataset in enumerate(self.datasets):
                    if positions[dataset_idx] < len(dataset):
                        self.index.append((dataset_idx, positions[dataset_idx]))
                        positions[dataset_idx] += 1
                        added = True
                        if len(self.index) >= max_samples:
                            break
                if not added:
                    break
        else:
            for dataset_idx, dataset in enumerate(self.datasets):
                for sample_idx in range(len(dataset)):
                    self.index.append((dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        dataset = self.datasets[dataset_idx]
        sample = dataset[sample_idx]
        sample_info = dataset.sample_info[sample_idx]
        return {
            "table": sample["measurement_table"],
            "task": sample_info["task"],
            "label": parse_binary_label(sample_info["target"]),
        }

    def task_names(self):
        tasks = set()
        for dataset_idx, sample_idx in self.index:
            tasks.add(str(self.datasets[dataset_idx].sample_info[sample_idx]["task"]))
        return sorted(tasks)


class TaskQueryCollator:
    def __init__(
        self,
        text_to_idx: dict[str, int],
        pad_idx: int,
        type_vocab: dict[str, int],
        query_embeddings: dict[str, torch.Tensor],
        max_table_len: Optional[int],
        min_table_rows: int,
    ):
        self.text_to_idx = text_to_idx
        self.pad_idx = pad_idx
        self.type_vocab = type_vocab
        self.query_embeddings = query_embeddings
        self.max_table_len = max_table_len
        self.min_table_rows = min_table_rows

    def __call__(self, batch):
        tables = []
        query_embeds = []
        labels = []

        for sample in batch:
            table = sample["table"]
            if table is None or table.empty:
                continue
            if self.max_table_len is not None:
                table = table.tail(self.max_table_len).reset_index(drop=True)
            if len(table) < self.min_table_rows:
                continue

            task_name = str(sample["task"])
            tables.append(table)
            query_embeds.append(self.query_embeddings[task_name])
            labels.append(sample["label"])

        if len(tables) == 0:
            raise ValueError("All samples in this batch have fewer than two table rows.")

        table_tensors = build_table_token_tensors(
            tables,
            text_to_idx=self.text_to_idx,
            pad_idx=self.pad_idx,
            type_vocab=self.type_vocab,
        )
        table_tensors["query_embeds"] = torch.stack(query_embeds)
        table_tensors["labels"] = torch.tensor(labels, dtype=torch.float)
        return table_tensors


def load_type_vocab(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def main():
    parser = HfArgumentParser((DataArguments, TrainingArgumentsCustom))
    data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    set_seed(training_args.seed)

    train_paths = resolve_sample_info_paths(data_args.train_sample_info_path)
    val_paths = resolve_sample_info_paths(data_args.val_sample_info_path)
    if len(train_paths) == 0:
        raise ValueError(f"No train sample_info files found: {data_args.train_sample_info_path}")
    if len(val_paths) == 0:
        raise ValueError(f"No val sample_info files found: {data_args.val_sample_info_path}")

    embedding_cache, text_dim = load_embedding_cache(data_args.table_text_embedding)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    special_indices = get_special_token_indices(text_to_idx)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)

    type_vocab = load_type_vocab(data_args.type_vocab_file)
    type_vocab_size = max(type_vocab.values()) + 1

    train_dataset = TaskQueryDataset(data_args.root_dir, train_paths, data_args.max_train_samples)
    val_dataset = TaskQueryDataset(data_args.root_dir, val_paths, data_args.max_eval_samples)
    task_names = sorted(set(train_dataset.task_names()) | set(val_dataset.task_names()))
    query_embeddings, query_dim = build_query_embeddings(
        task_names=task_names,
        cache_path=data_args.query_embedding_cache,
        model_path=data_args.query_text_encoder_path,
        base_model_path=data_args.query_text_encoder_base_model,
        max_length=data_args.query_max_length,
    )

    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=type_vocab_size,
        max_table_len=data_args.max_table_len,
        num_classes=1,
        problem_type="single_label_classification",
    )
    model = TaskQueryClassificationModel(
        config=config,
        embedding_matrix=embedding_matrix,
        query_dim=query_dim,
    )
    model = load_model_weights(model, data_args.pretrained_path)

    collator = TaskQueryCollator(
        text_to_idx=text_to_idx,
        pad_idx=special_indices["pad_idx"],
        type_vocab=type_vocab,
        query_embeddings=query_embeddings,
        max_table_len=data_args.max_table_len,
        min_table_rows=data_args.min_table_rows,
    )

    rank0_print(f"Train files: {len(train_paths)}")
    rank0_print(f"Val files: {len(val_paths)}")
    rank0_print(f"Train samples: {len(train_dataset)}")
    rank0_print(f"Val samples: {len(val_dataset)}")
    rank0_print(f"Tasks: {', '.join(task_names)}")
    rank0_print(f"Table text_dim={text_dim}, query_dim={query_dim}, type_vocab_size={type_vocab_size}")

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
        )
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_classification_metrics,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
