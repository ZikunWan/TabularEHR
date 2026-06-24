import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info as get_ehrshot_task_info
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info as get_eicu_task_info
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info as get_mimic_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.query_classifier import TaskQueryClassificationModel
from utils.collate import build_table_token_tensors
from utils.load_embedding import build_text_to_idx
from utils.metrics import compute_classification_metrics
from utils.weight_loader import load_encoder_weights


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
    dataset: List[str] = field(default_factory=lambda: ["mimic_iv"])
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    train_sample_info_path: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train")
    val_sample_info_path: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val")
    table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings.pt"]
    )
    eicu_root_dir: str = field(default="/data/zikun_workspace/eicu-crd")
    eicu_processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    eicu_train_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json")
    eicu_val_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json")
    eicu_table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt"]
    )
    ehrshot_root_dir: str = field(default="/data/EHR_data_public/EHRSHOT")
    ehrshot_train_sample_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv")
    ehrshot_val_sample_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv")
    ehrshot_table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt"]
    )
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    pretrained_path: Optional[str] = field(default="/data/zikun_workspace/checkpoints/pretraining/contrastive_learning")
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/task_query_classification/task_query_llm_embeddings.pt")
    query_llm_model_path: str = field(default="/data/model_weights_public/BlueZeros/EHR-R1-1.7B")
    query_max_length: int = field(default=512)
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


def resolve_task_names_from_paths(sample_info_paths):
    return sorted(
        {
            os.path.splitext(os.path.basename(sample_info_path))[0]
            for sample_info_path in sample_info_paths
        }
    )


def get_task_info():
    task_info = {}
    task_info.update(get_mimic_task_info())
    task_info.update(get_eicu_task_info())
    task_info.update(get_ehrshot_task_info())
    return task_info


def binary_task_names(task_info: dict):
    return sorted(
        task_name
        for task_name, info in task_info.items()
        if info["task_type"] == "binary_classification"
    )


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

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()

    if tokenizer.chat_template:
        query_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": query_text}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for query_text in query_texts
        ]

    tokens = tokenizer(
        query_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        last_indices = get_llm_pooling_indices(
            model_path,
            tokenizer,
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        query_embeds = last_hidden[
            torch.arange(last_hidden.size(0), device=last_hidden.device),
            last_indices,
        ].cpu().to(torch.bfloat16)

    embeddings = {
        task_name: query_embeds[idx]
        for idx, task_name in enumerate(task_names)
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": int(query_embeds.size(-1)),
            "model_path": model_path,
        },
        cache_path,
    )
    rank0_print(f"Saved query embeddings to {cache_path}")
    return embeddings, int(query_embeds.size(-1))


def get_llm_pooling_indices(model_path: str, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    model_name_lower = model_path.lower()
    eos_id = None
    if "qwen" in model_name_lower or "ehr-r1" in model_name_lower:
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    if eos_id is not None:
        eos_mask = input_ids == eos_id
        if eos_mask.sum().item() > 0:
            seq_len = input_ids.size(1)
            indices = torch.arange(seq_len, device=input_ids.device)
            return (eos_mask * indices).argmax(dim=1)

    return attention_mask.sum(dim=1) - 1


def parse_binary_label(value) -> int:
    label = str(value).strip().strip('"').strip("'").strip().lower()
    if label == "true":
        return 1
    if label == "false":
        return 0
    if label == "yes":
        return 1
    if label == "no":
        return 0
    return int(float(label))


def load_table_embeddings(cache_paths: List[str]):
    embedding_cache = {}
    for cache_path in cache_paths:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        embedding_cache.update(data["embeddings"])
        text_dim = int(data["text_dim"])
        rank0_print(f"Loaded {len(data['embeddings'])} embeddings from {cache_path}")

    vocab_keys = list(embedding_cache.keys())
    text_to_idx = build_text_to_idx(vocab_keys)
    matrix = torch.empty(len(vocab_keys), text_dim)
    for idx, text in enumerate(vocab_keys):
        matrix[idx] = embedding_cache[text]
    return text_dim, vocab_keys, text_to_idx, matrix


class TaskQueryDataset(Dataset):
    def __init__(self, datasets, max_samples: Optional[int]):
        self.datasets = datasets
        self.index = []
        if max_samples is not None:
            positions = [0] * len(self.datasets)
            while len(self.index) < max_samples:
                added = False
                for dataset_idx, (dataset_name, dataset) in enumerate(self.datasets):
                    if positions[dataset_idx] < len(dataset):
                        self.index.append((dataset_idx, positions[dataset_idx]))
                        positions[dataset_idx] += 1
                        added = True
                        if len(self.index) >= max_samples:
                            break
                if not added:
                    break
        else:
            for dataset_idx, (_, dataset) in enumerate(self.datasets):
                for sample_idx in range(len(dataset)):
                    self.index.append((dataset_idx, sample_idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self.index[idx]
        dataset_name, dataset = self.datasets[dataset_idx]
        sample = dataset[sample_idx]
        sample_info = dataset.sample_info[sample_idx]
        if dataset_name == "mimic_iv":
            task_name = sample_info["task"]
            label = sample_info["target"]
        else:
            task_name = sample_info["task_name"]
            label = sample["output"]
        return {
            "table": sample["measurement_table"],
            "task": task_name,
            "label": parse_binary_label(label),
        }

    def task_names(self):
        tasks = set()
        for dataset_idx, sample_idx in self.index:
            dataset_name, dataset = self.datasets[dataset_idx]
            sample_info = dataset.sample_info[sample_idx]
            if dataset_name == "mimic_iv":
                tasks.add(str(sample_info["task"]))
            else:
                tasks.add(str(sample_info["task_name"]))
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


def build_mimic_datasets(root_dir: str, sample_info_paths):
    return [
        (
            "mimic_iv",
            MIMICIV(
                root_dir=root_dir,
                sample_info_path=sample_info_path,
                lazy_mode=True,
                shuffle=False,
                max_samples=None,
                use_table_length_cache=False,
            ),
        )
        for sample_info_path in sample_info_paths
    ]


def build_eicu_datasets(root_dir: str, processed_dir: str, sample_info, task_names):
    return [
        (
            "eicu",
            EICUDataset(
                root_dir=root_dir,
                processed_dir=processed_dir,
                sample_info=sample_info,
                task_name=task_name,
                lazy_mode=True,
                shuffle=False,
            ),
        )
        for task_name in task_names
    ]


def build_ehrshot_datasets(root_dir: str, sample_info, task_names):
    return [
        (
            "ehrshot",
            EHRSHOTDataset(
                root_dir=root_dir,
                sample_info=sample_info,
                task_name=task_name,
                lazy_mode=True,
            ),
        )
        for task_name in task_names
    ]


def load_json_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_records(path: str):
    return pd.read_csv(path, low_memory=False).to_dict(orient="records")


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


def main():
    parser = HfArgumentParser((DataArguments, TrainingArgumentsCustom))
    data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    set_seed(training_args.seed)

    task_info = get_task_info()
    binary_tasks = binary_task_names(task_info)
    train_paths = resolve_sample_info_paths(data_args.train_sample_info_path)
    val_paths = resolve_sample_info_paths(data_args.val_sample_info_path)
    if "mimic_iv" in data_args.dataset and len(train_paths) == 0:
        raise ValueError(f"No train sample_info files found: {data_args.train_sample_info_path}")
    if "mimic_iv" in data_args.dataset and len(val_paths) == 0:
        raise ValueError(f"No val sample_info files found: {data_args.val_sample_info_path}")

    train_dataset_parts = []
    val_dataset_parts = []
    if "mimic_iv" in data_args.dataset:
        train_dataset_parts.extend(build_mimic_datasets(data_args.root_dir, train_paths))
        val_dataset_parts.extend(build_mimic_datasets(data_args.root_dir, val_paths))
    if "eicu" in data_args.dataset:
        eicu_tasks = [task_name for task_name in binary_tasks if task_name in get_eicu_task_info()]
        eicu_train_sample_info = load_json_records(data_args.eicu_train_sample_info_path)
        eicu_val_sample_info = load_json_records(data_args.eicu_val_sample_info_path)
        train_dataset_parts.extend(build_eicu_datasets(data_args.eicu_root_dir, data_args.eicu_processed_dir, eicu_train_sample_info, eicu_tasks))
        val_dataset_parts.extend(build_eicu_datasets(data_args.eicu_root_dir, data_args.eicu_processed_dir, eicu_val_sample_info, eicu_tasks))
    if "ehrshot" in data_args.dataset:
        ehrshot_tasks = [task_name for task_name in binary_tasks if task_name in get_ehrshot_task_info()]
        ehrshot_train_sample_info = load_csv_records(data_args.ehrshot_train_sample_info_path)
        ehrshot_val_sample_info = load_csv_records(data_args.ehrshot_val_sample_info_path)
        train_dataset_parts.extend(build_ehrshot_datasets(data_args.ehrshot_root_dir, ehrshot_train_sample_info, ehrshot_tasks))
        val_dataset_parts.extend(build_ehrshot_datasets(data_args.ehrshot_root_dir, ehrshot_val_sample_info, ehrshot_tasks))

    train_dataset = TaskQueryDataset(train_dataset_parts, data_args.max_train_samples)
    val_dataset = TaskQueryDataset(val_dataset_parts, data_args.max_eval_samples)
    task_names = sorted(set(train_dataset.task_names()) | set(val_dataset.task_names()))
    query_embeddings, query_dim = build_query_embeddings(
        task_names=task_names,
        cache_path=data_args.query_embedding_cache,
        model_path=data_args.query_llm_model_path,
        max_length=data_args.query_max_length,
    )

    text_dim, vocab_keys, text_to_idx, embedding_matrix = load_table_embeddings(get_embedding_cache_paths(data_args))

    type_vocab = load_type_vocab(data_args.type_vocab_file)
    type_vocab_size = max(type_vocab.values()) + 1

    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=type_vocab_size,
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        num_classes=1,
        problem_type="single_label_classification",
    )
    model = TaskQueryClassificationModel(
        config=config,
        embedding_matrix=embedding_matrix,
        query_dim=query_dim,
    )
    model = load_encoder_weights(model, data_args.pretrained_path)

    collator = TaskQueryCollator(
        text_to_idx=text_to_idx,
        pad_idx=0,
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
