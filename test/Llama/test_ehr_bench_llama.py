import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic.mimic_dataset import MIMICIV
from train.Llama.train_ehr_bench_llama import (
    ALL_RISK_PREDICTION_TASKS,
    EHRBenchMEDSDataCollator,
    _build_label_metadata,
)
from train.Llama.train_ehrshot_llama import (
    CLASSIFICATION_HEAD_METADATA_FILENAME,
    LlamaMEDSClassifier,
    _load_clmbr_tokenizer,
    load_sequence_classifier_checkpoint,
    rank0_print,
)

TRAIN_METADATA_FILENAME = "sequence_classification_metadata.json"


@dataclass
class ModelArguments:
    checkpoint_dir: str = field(metadata={"help": "Path to fine-tuned classifier checkpoint directory."})


@dataclass
class DataArguments:
    sample_info_path: str = field(metadata={"help": "Path to test CSV."})
    output_dir: str = field(metadata={"help": "Directory for evaluation outputs."})
    data_dir: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular",
        metadata={"help": "Root directory for MIMIC-IV tabular data used by EHR-Bench."},
    )
    task_name: str = field(
        default="ED_Hospitalization",
        metadata={"help": f"EHR-Bench risk task. One of: {ALL_RISK_PREDICTION_TASKS}"},
    )
    max_samples: Optional[int] = field(default=None, metadata={"help": "Maximum test samples."})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum tokenized sequence length."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    lazy_mode: bool = field(default=True, metadata={"help": "Load test samples lazily."})
    itemid_representation: str = field(
        default="code",
        metadata={"help": "MEDS code representation. Use 'code' or 'description'."},
    )
    concept_map_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional concept map directory when itemid_representation='code'."},
    )


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_sequence_classification_metrics(logits, labels, task_name: str, idx_list=None, id2label: Optional[dict] = None):
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim != 2:
        raise ValueError(f"logits must be a 2D array, got shape={logits.shape}")
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if len(logits) != len(labels):
        raise ValueError(f"logits and labels size mismatch: {len(logits)} vs {len(labels)}")

    shifted_logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    if probs.shape[1] == 2:
        auroc = roc_auc_score(labels, probs[:, 1])
    else:
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    metrics_df = pd.DataFrame(
        [
            {
                "Task": task_name,
                "Count": int(len(labels)),
                "Metric": "Macro AUROC" if probs.shape[1] > 2 else "AUROC",
                "Value": float(auroc),
                "Accuracy": float(acc),
                "Macro F1": float(macro_f1),
            }
        ]
    )

    raw_df = pd.DataFrame(
        {
            "idx": list(idx_list),
            "label": labels.tolist(),
            "pred": preds.tolist(),
            "prob": probs.tolist(),
        }
    )
    if id2label is not None:
        raw_df["label_name"] = [id2label[int(label)] for label in labels]
        raw_df["pred_name"] = [id2label[int(pred)] for pred in preds]

    return metrics_df, raw_df


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    if data_args.task_name not in ALL_RISK_PREDICTION_TASKS:
        raise ValueError(
            f"Unsupported task_name '{data_args.task_name}'. "
            f"Supported tasks: {ALL_RISK_PREDICTION_TASKS}"
        )
    set_seed(data_args.seed)

    head_metadata = _load_json(os.path.join(model_args.checkpoint_dir, CLASSIFICATION_HEAD_METADATA_FILENAME))
    train_metadata = _load_json(os.path.join(model_args.checkpoint_dir, TRAIN_METADATA_FILENAME))
    base_model_name_or_path = train_metadata["model_name_or_path"]

    candidates, label_to_id, id_to_label = _build_label_metadata(data_args.task_name)
    if train_metadata["task_name"] != data_args.task_name:
        raise ValueError(
            f"Task mismatch: checkpoint was trained for '{train_metadata['task_name']}' "
            f"but evaluation task is '{data_args.task_name}'."
        )
    checkpoint_num_labels = int(head_metadata["num_labels"])
    if checkpoint_num_labels != len(candidates):
        raise ValueError(
            f"Label count mismatch: checkpoint expects {checkpoint_num_labels} labels, "
            f"but task '{data_args.task_name}' has {len(candidates)} labels."
        )

    rank0_print("=" * 80)
    rank0_print("EHR-Bench MEDS Llama Encoder Classifier Test")
    rank0_print("=" * 80)
    rank0_print(f"Checkpoint directory: {model_args.checkpoint_dir}")
    rank0_print(f"Base model path: {base_model_name_or_path}")
    rank0_print(f"Task: {data_args.task_name}")
    rank0_print(f"Max seq length: {data_args.max_seq_length}")
    rank0_print(f"ItemID representation: {data_args.itemid_representation}")
    rank0_print(f"Concept map dir: {data_args.concept_map_dir}")

    tokenizer = _load_clmbr_tokenizer(model_args.checkpoint_dir)
    model = LlamaMEDSClassifier(
        model_name_or_path=base_model_name_or_path,
        num_labels=len(candidates),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        freeze_encoder=bool(train_metadata.get("freeze_encoder", True)),
        tokenizer_vocab_size=int(tokenizer.vocab_size),
        use_peft=False,
    )
    model = load_sequence_classifier_checkpoint(model, model_args.checkpoint_dir, train_metadata)

    eval_dataset = MIMICIV(
        root_dir=data_args.data_dir,
        sample_info_path=data_args.sample_info_path,
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode="text_only",
        max_samples=data_args.max_samples,
        itemid_representation=data_args.itemid_representation,
        concept_map_dir=data_args.concept_map_dir,
        return_meds=True,
    )
    rank0_print(f"test source [{data_args.task_name}, MEDS] size: {len(eval_dataset)}")

    data_collator = EHRBenchMEDSDataCollator(
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_seq_length=data_args.max_seq_length,
        task_name=data_args.task_name,
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

    idx_list = list(range(len(predict_outputs.label_ids)))
    metrics_df, raw_df = _compute_sequence_classification_metrics(
        logits=predict_outputs.predictions,
        labels=predict_outputs.label_ids,
        task_name=data_args.task_name,
        idx_list=idx_list,
        id2label=id_to_label,
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
