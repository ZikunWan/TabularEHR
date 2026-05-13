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

from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info
from train.Llama.train_ehrshot_llama import (
    CLASSIFICATION_HEAD_METADATA_FILENAME,
    CLASSIFICATION_HEAD_STATE_FILENAME,
    LlamaMEDSClassifier,
    _load_clmbr_tokenizer,
    rank0_print,
)
from train.Llama.train_eicu_llama import (
    EICUMEDSDataCollator,
    _build_label_metadata,
)

TRAIN_METADATA_FILENAME = "sequence_classification_metadata.json"


@dataclass
class ModelArguments:
    checkpoint_dir: str = field(metadata={"help": "Path to fine-tuned classifier checkpoint directory."})


@dataclass
class DataArguments:
    output_dir: str = field(metadata={"help": "Directory for evaluation outputs."})
    root_dir: str = field(
        default="/home/ma-user/sfs_turbo/Data/eicu-crd/2.0",
        metadata={"help": "Root directory for raw eICU data."},
    )
    processed_dir: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed",
        metadata={"help": "Processed eICU directory."},
    )
    sample_info_test_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_test.json")
    task_name: str = field(default="mortality", metadata={"help": "Single eICU task name."})
    max_samples: Optional[int] = field(default=None, metadata={"help": "Maximum test samples."})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum tokenized sequence length."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    lazy_mode: bool = field(default=True, metadata={"help": "Load samples lazily."})
    table_mode: str = field(
        default="text_only",
        metadata={"help": "Input mode: text_only/table_only/table_plus_rest_text."},
    )


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_classifier_state_dict(checkpoint_dir: str):
    head_path = os.path.join(checkpoint_dir, CLASSIFICATION_HEAD_STATE_FILENAME)
    state_dict = torch.load(head_path, map_location="cpu")
    return {key[len("classifier."):]: value for key, value in state_dict.items()}


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

    metric_name = get_task_info()[task_name]["metric"]
    if metric_name == "accuracy":
        main_metric = "Accuracy"
        main_value = float(acc)
    else:
        main_metric = "Macro AUROC" if probs.shape[1] > 2 else "AUROC"
        main_value = float(auroc)

    metrics_df = pd.DataFrame(
        [
            {
                "Task": task_name,
                "Count": int(len(labels)),
                "Metric": main_metric,
                "Value": main_value,
                "AUROC": float(auroc),
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
    raw_df["label_name"] = [id2label[int(label)] for label in labels]
    raw_df["pred_name"] = [id2label[int(pred)] for pred in preds]

    return metrics_df, raw_df


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    task_schema = get_task_info()
    if data_args.task_name not in task_schema:
        raise ValueError(f"Unsupported eICU task_name '{data_args.task_name}'.")
    if task_schema[data_args.task_name]["task_type"] == "multi_label_classification":
        raise NotImplementedError(
            f"MEDS encoder testing does not currently support multi-label eICU task '{data_args.task_name}'."
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
    rank0_print("eICU MEDS Llama Encoder Classifier Test")
    rank0_print("=" * 80)
    rank0_print(f"Checkpoint directory: {model_args.checkpoint_dir}")
    rank0_print(f"Base model path: {base_model_name_or_path}")
    rank0_print(f"Task: {data_args.task_name}")
    rank0_print(f"Max seq length: {data_args.max_seq_length}")

    tokenizer = _load_clmbr_tokenizer(model_args.checkpoint_dir)
    model = LlamaMEDSClassifier(
        model_name_or_path=base_model_name_or_path,
        num_labels=len(candidates),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        freeze_encoder=True,
        tokenizer_vocab_size=int(tokenizer.vocab_size),
    )
    model.classifier.load_state_dict(_load_classifier_state_dict(model_args.checkpoint_dir))

    sample_info_test_path = data_args.sample_info_test_path

    eval_dataset = EICUDataset(
        root_dir=data_args.root_dir,
        processed_dir=data_args.processed_dir,
        sample_info_path=sample_info_test_path,
        task_name=data_args.task_name,
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode=data_args.table_mode,
        max_samples=data_args.max_samples,
        return_meds=True,
    )
    rank0_print(f"test source [{data_args.task_name}, MEDS] size: {len(eval_dataset)}")

    data_collator = EICUMEDSDataCollator(
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
