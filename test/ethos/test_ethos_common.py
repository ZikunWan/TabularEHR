import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "train" / "ethos"))

from train_ethos_common import EthosOnTheFlyCollator, build_ethos_model, build_label_metadata, rank0_print


@dataclass
class EthosTestArguments:
    checkpoint_dir: str = field(metadata={"help": "ETHOS checkpoint directory."})
    vocab_dir: Optional[str] = field(default=None)
    max_seq_length: int = field(default=4096)
    batch_size: int = field(default=16)
    seed: int = field(default=42)
    output_dir: Optional[str] = field(default=None)
    n_layer: Optional[int] = field(default=None)
    n_head: Optional[int] = field(default=None)
    n_embd: Optional[int] = field(default=None)
    dropout: Optional[float] = field(default=None)


def load_metadata(checkpoint_dir):
    path = os.path.join(checkpoint_dir, "ethos_training_metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def model_args_from_metadata(args, metadata):
    return SimpleNamespace(
        model_name_or_path=None,
        freeze_encoder=bool(metadata.get("freeze_encoder", False)),
        n_layer=int(args.n_layer or metadata["n_layer"]),
        n_head=int(args.n_head or metadata["n_head"]),
        n_embd=int(args.n_embd or metadata["n_embd"]),
        dropout=float(args.dropout if args.dropout is not None else 0.0),
        classifier_dropout=None,
    )


def load_state_dict(checkpoint_dir):
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path, device="cpu")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No ETHOS weights found in {checkpoint_dir}")


def compute_metrics_df(logits, labels, task_name, id_to_label):
    logits = np.asarray(logits)
    labels = np.asarray(labels).reshape(-1).astype(int)
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1).astype(int)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(labels, probs[:, 1]) if probs.shape[1] == 2 else roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.5

    metrics_df = pd.DataFrame([{
        "Task": task_name,
        "Count": int(len(labels)),
        "Metric": "Macro AUROC" if probs.shape[1] > 2 else "AUROC",
        "Value": float(auroc),
        "AUROC": float(auroc),
        "Accuracy": float(acc),
        "Macro F1": float(macro_f1),
    }])
    raw_df = pd.DataFrame({
        "idx": list(range(len(labels))),
        "label": labels.tolist(),
        "pred": preds.tolist(),
        "prob": probs.tolist(),
        "label_name": [id_to_label.get(int(x), x) for x in labels],
        "pred_name": [id_to_label.get(int(x), x) for x in preds],
    })
    return metrics_df, raw_df


def run_ethos_test(*, args, task_name, task_info, eval_dataset):
    set_seed(args.seed)
    metadata = load_metadata(args.checkpoint_dir)
    vocab_dir = args.vocab_dir or metadata["vocab_dir"]
    max_seq_length = int(args.max_seq_length or metadata["max_seq_length"])

    candidates, label_to_id, id_to_label = build_label_metadata(task_info, task_name)
    model_args = model_args_from_metadata(args, metadata)
    model, vocab = build_ethos_model(
        model_args,
        vocab_dir=vocab_dir,
        max_seq_length=max_seq_length,
        num_labels=len(candidates),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
    )
    model.load_state_dict(load_state_dict(args.checkpoint_dir), strict=False)

    collator = EthosOnTheFlyCollator(
        vocab_dir=vocab_dir,
        label_to_id=label_to_id,
        task_name=task_name,
        max_seq_length=max_seq_length,
    )
    output_dir = args.output_dir or os.path.join(args.checkpoint_dir, "eval_logs")

    rank0_print("=" * 80)
    rank0_print(f"ETHOS Test: {task_name}")
    rank0_print("=" * 80)
    rank0_print(f"Checkpoint: {args.checkpoint_dir}")
    rank0_print(f"Vocab dir: {vocab_dir}")
    rank0_print(f"Vocab size: {len(vocab)}")
    rank0_print(f"Test size: {len(eval_dataset)}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=args.batch_size,
            remove_unused_columns=False,
            report_to="none",
        ),
        data_collator=collator,
    )
    outputs = trainer.predict(eval_dataset)
    if not trainer.is_world_process_zero():
        return

    metrics_df, raw_df = compute_metrics_df(outputs.predictions, outputs.label_ids, task_name, id_to_label)
    rank0_print(metrics_df.to_string(index=False))
    os.makedirs(output_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    raw_df.to_csv(os.path.join(output_dir, "raw_predictions.csv"), index=False)


def parse_args(data_cls):
    return HfArgumentParser((EthosTestArguments, data_cls)).parse_args_into_dataclasses()
