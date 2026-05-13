import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.Llama.train_ehrshot_llama import (
    CLASSIFICATION_HEAD_METADATA_FILENAME,
    CLASSIFICATION_HEAD_STATE_FILENAME,
    _load_clmbr_tokenizer,
    rank0_print,
)
from train.Llama.train_renji_llama import (
    RenjiLlamaMEDSMultiLabelClassifier,
    RenjiMEDSDataCollator,
    RenjiMEDSDataset,
    _build_label_metadata,
    _parse_csv_arg,
)
from utils.metrics import calc_accuracy, calc_auroc, calc_f1, calc_recall

TRAIN_METADATA_FILENAME = "sequence_classification_metadata.json"


@dataclass
class ModelArguments:
    checkpoint_dir: str = field(metadata={"help": "Path to fine-tuned Renji classifier checkpoint directory."})


@dataclass
class DataArguments:
    output_dir: str = field(metadata={"help": "Directory for evaluation outputs."})
    root_dir: str = field(default="/data/EHR_data_public/Renji")
    split: str = field(default="test")
    target_prediction_points: str = field(
        default="day0,day30,day180,day365",
        metadata={"help": "Comma-separated Renji prediction points, e.g. day30,day180."},
    )
    max_samples: Optional[int] = field(default=None)
    max_seq_length: int = field(default=4096)
    batch_size: int = field(default=16)
    seed: int = field(default=42)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_classifier_head_state(checkpoint_dir: str):
    head_path = os.path.join(checkpoint_dir, CLASSIFICATION_HEAD_STATE_FILENAME)
    state_dict = torch.load(head_path, map_location="cpu")
    return {key[len("classifier."):]: value for key, value in state_dict.items()}


def _compute_multilabel_outputs(logits, labels, label_names):
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    mask = labels != -100

    y_true = labels[mask].astype(int)
    y_prob = probs[mask]
    y_pred = preds[mask]

    metrics_df = pd.DataFrame(
        [
            {
                "Task": "renji_multi_label_prediction",
                "Count": int(y_true.shape[0]),
                "Metric": "AUROC",
                "Value": float(calc_auroc(y_true, y_prob)),
                "AUROC": float(calc_auroc(y_true, y_prob)),
                "Accuracy": float(calc_accuracy(y_true, y_pred)),
                "F1": float(calc_f1(y_true, y_pred, average="binary")),
                "Recall": float(calc_recall(y_true, y_pred, average="binary")),
            }
        ]
    )

    rows = []
    for sample_idx in range(labels.shape[0]):
        for label_idx, label_name in enumerate(label_names):
            if labels[sample_idx, label_idx] == -100:
                continue
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "label_idx": label_idx,
                    "label_name": label_name,
                    "label": int(labels[sample_idx, label_idx]),
                    "pred": int(preds[sample_idx, label_idx]),
                    "prob": float(probs[sample_idx, label_idx]),
                    "logit": float(logits[sample_idx, label_idx]),
                }
            )
    raw_df = pd.DataFrame(rows)
    return metrics_df, raw_df


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    set_seed(data_args.seed)

    head_metadata = _load_json(os.path.join(model_args.checkpoint_dir, CLASSIFICATION_HEAD_METADATA_FILENAME))
    train_metadata = _load_json(os.path.join(model_args.checkpoint_dir, TRAIN_METADATA_FILENAME))
    base_model_name_or_path = train_metadata["model_name_or_path"]

    label_names, label_to_id, id_to_label = _build_label_metadata()
    if train_metadata["task_name"] != "renji_multi_label_prediction":
        raise ValueError(f"Checkpoint task is {train_metadata['task_name']}, expected renji_multi_label_prediction.")
    if int(head_metadata["num_labels"]) != len(label_names):
        raise ValueError(f"Checkpoint has {head_metadata['num_labels']} labels, expected {len(label_names)}.")

    rank0_print("=" * 80)
    rank0_print("Renji MEDS Llama Encoder Multi-Label Test")
    rank0_print("=" * 80)
    rank0_print(f"Checkpoint directory: {model_args.checkpoint_dir}")
    rank0_print(f"Base model path: {base_model_name_or_path}")
    rank0_print(f"Split: {data_args.split}")
    rank0_print(f"Target prediction points: {data_args.target_prediction_points}")
    rank0_print(f"Max seq length: {data_args.max_seq_length}")

    tokenizer = _load_clmbr_tokenizer(model_args.checkpoint_dir)
    model = RenjiLlamaMEDSMultiLabelClassifier(
        model_name_or_path=base_model_name_or_path,
        num_labels=len(label_names),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        freeze_encoder=True,
        tokenizer_vocab_size=int(tokenizer.vocab_size),
    )
    model.classifier.load_state_dict(_load_classifier_head_state(model_args.checkpoint_dir))

    eval_dataset = RenjiMEDSDataset(
        root_dir=data_args.root_dir,
        split=data_args.split,
        target_prediction_points=_parse_csv_arg(data_args.target_prediction_points),
        shuffle=False,
        max_samples=data_args.max_samples,
    )
    rank0_print(f"{data_args.split} source [Renji multi-label, MEDS] size: {len(eval_dataset)}")

    data_collator = RenjiMEDSDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )

    output_dir = data_args.output_dir
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=data_args.batch_size,
            remove_unused_columns=False,
            report_to="none",
            bf16=True,
            fp16=False,
        ),
        data_collator=data_collator,
    )

    rank0_print("Starting evaluation...")
    predict_outputs = trainer.predict(eval_dataset)
    if not trainer.is_world_process_zero():
        return

    metrics_df, raw_df = _compute_multilabel_outputs(
        logits=predict_outputs.predictions,
        labels=predict_outputs.label_ids,
        label_names=label_names,
    )
    rank0_print(metrics_df.to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    raw_path = os.path.join(output_dir, "raw_predictions.csv")
    metrics_df.to_csv(metrics_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    rank0_print(f"Metrics saved to {metrics_path}")
    rank0_print(f"Raw predictions saved to {raw_path}")


if __name__ == "__main__":
    main()
