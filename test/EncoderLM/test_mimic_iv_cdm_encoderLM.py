import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import (
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import build_prompts_from_dataset, detect_model_family, load_model, load_tokenizer
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info


MAIN_DIAGNOSIS_TASK = "MIMIC-IV-CDM Main Disease Diagnoses"


@dataclass
class ModelArguments:
    checkpoint_dir: str = field(metadata={"help": "Path to the fine-tuned checkpoint directory."})


@dataclass
class DataArguments:
    root_dir: str = field(
        default="/home/ma-user/sfs_turbo/Data/mimic-iv-cdm",
        metadata={"help": "Root directory for MIMIC-IV-CDM data."},
    )
    task_name: str = field(
        default=MAIN_DIAGNOSIS_TASK,
        metadata={"help": "Currently only supports 'MIMIC-IV-CDM Main Disease Diagnoses'."},
    )
    table_mode: str = field(
        default="table_only",
        metadata={"help": "Input mode: 'text_only', 'table_only', or 'table_plus_rest_text'."},
    )
    lazy_mode: bool = field(default=False, metadata={"help": "Load MIMIC-IV-CDM samples lazily."})
    max_seq_len: int = field(default=8192, metadata={"help": "Maximum context length."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    output_dir: Optional[str] = field(default=None, metadata={"help": "Directory for evaluation outputs."})


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    candidates = list(task_info["candidate"])
    label_to_id = {label: idx for idx, label in enumerate(candidates)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return candidates, label_to_id, id_to_label


def _tokenize_prompts(prompts, meta_list, tokenizer, label_to_id, max_seq_len: int):
    rows = []
    for prompt, meta in zip(prompts, meta_list):
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_len,
        )
        tokenized["labels"] = label_to_id[str(meta["label"])]
        tokenized["idx"] = meta["idx"]
        rows.append(tokenized)
    from datasets import Dataset
    return Dataset.from_list(rows)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    if data_args.task_name != MAIN_DIAGNOSIS_TASK:
        raise NotImplementedError(
            "EncoderLM testing currently supports only 'MIMIC-IV-CDM Main Disease Diagnoses'."
        )
    set_seed(data_args.seed)

    if not os.path.isdir(model_args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {model_args.checkpoint_dir}")

    candidates, label_to_id, id_to_label = _build_label_metadata(data_args.task_name)

    print("=" * 80)
    print("MIMIC-IV-CDM EncoderLM Test")
    print("=" * 80)
    print(f"Checkpoint directory: {model_args.checkpoint_dir}")
    print(f"Task: {data_args.task_name}")
    print(f"Table mode: {data_args.table_mode}")
    print(f"Max seq len: {data_args.max_seq_len}")

    model = load_model(
        model_args.checkpoint_dir,
        num_labels=len(candidates),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    tokenizer = load_tokenizer(model_args.checkpoint_dir)
    detect_model_family(model_args.checkpoint_dir)

    val_dataset = MIMICIVCDM(
        root_dir=data_args.root_dir,
        split="val",
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode=data_args.table_mode,
        task_name=data_args.task_name,
        max_samples=None,
    )
    test_dataset = MIMICIVCDM(
        root_dir=data_args.root_dir,
        split="test",
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode=data_args.table_mode,
        task_name=data_args.task_name,
        max_samples=None,
    )
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    test_dataset.list_data = val_dataset.list_data + test_dataset.list_data
    if hasattr(test_dataset, "data") and hasattr(val_dataset, "data"):
        test_dataset.data = val_dataset.data + test_dataset.data
    dataset = test_dataset

    print(
        f"merged eval source [val+test, {data_args.task_name}, {data_args.table_mode}] "
        f"size: {len(dataset)} (val={val_size}, test={test_size})"
    )

    prompts, meta_list = build_prompts_from_dataset(
        dataset,
        tokenizer,
        system_prompt="",
        max_seq_length=data_args.max_seq_len,
    )
    eval_dataset = _tokenize_prompts(prompts, meta_list, tokenizer, label_to_id, data_args.max_seq_len)

    output_dir = data_args.output_dir or os.path.join(model_args.checkpoint_dir, "eval_logs")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=data_args.batch_size,
            remove_unused_columns=False,
            report_to="none",
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Starting evaluation...")
    predict_outputs = trainer.predict(eval_dataset)
    logits = predict_outputs.predictions
    labels_np = predict_outputs.label_ids

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)

    try:
        auroc = roc_auc_score(labels_np, probs, multi_class="ovr")
    except ValueError:
        auroc = 0.5
    acc = accuracy_score(labels_np, preds)

    print(f"Task: {data_args.task_name}")
    print(f"Macro AUROC: {auroc:.4f}")
    print(f"Accuracy: {acc:.4f}")

    metrics_df = pd.DataFrame(
        [
            {
                "Task": data_args.task_name,
                "Count": len(labels_np),
                "Metric": "Macro AUROC",
                "Value": auroc,
                "Accuracy": acc,
            }
        ]
    )
    raw_df = pd.DataFrame(
        {
            "idx": [meta["idx"] for meta in meta_list],
            "label": labels_np.tolist(),
            "pred": preds.tolist(),
            "prob": probs.tolist(),
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    raw_path = os.path.join(output_dir, "raw_predictions.csv")
    metrics_df.to_csv(metrics_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    print(f"Raw predictions saved to {raw_path}")


if __name__ == "__main__":
    main()
