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
from train.MEDS_encoder.train_ehrshot_llama import (
    CLASSIFICATION_HEAD_METADATA_FILENAME,
    CLASSIFICATION_HEAD_STATE_FILENAME,
    LlamaMEDSClassifier,
    _load_clmbr_tokenizer,
    rank0_print,
)
from train.MEDS_encoder.train_eicu_llama import (
    EICUMEDSDataCollator,
    _build_label_metadata,
)

TRAIN_METADATA_FILENAME = "sequence_classification_metadata.json"


@dataclass
class ModelArguments:
    checkpoint_dir: str = field(metadata={"help": "Path to fine-tuned classifier checkpoint directory."})
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional base encoder path override. If unset, read from checkpoint metadata."},
    )


@dataclass
class DataArguments:
    root_dir: str = field(
        default="/home/ma-user/sfs_turbo/Data/eicu-crd/2.0",
        metadata={"help": "Root directory for raw eICU data."},
    )
    processed_dir: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed",
        metadata={"help": "Processed eICU directory."},
    )
    sample_info_test_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test sample_info JSON. Defaults to <processed_dir>/sample_info_test.json."},
    )
    task_name: str = field(default="mortality", metadata={"help": "Single eICU task name."})
    max_samples: Optional[int] = field(default=None, metadata={"help": "Maximum test samples."})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum tokenized sequence length."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    output_dir: Optional[str] = field(default=None, metadata={"help": "Directory for evaluation outputs."})
    lazy_mode: bool = field(default=True, metadata={"help": "Load samples lazily."})
    table_mode: str = field(
        default="text_only",
        metadata={"help": "Input mode: text_only/table_only/table_plus_rest_text."},
    )


def _load_json_if_exists(path: str):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _resolve_base_model_path(model_args: ModelArguments, head_metadata: Optional[dict], train_metadata: Optional[dict]):
    if model_args.model_name_or_path:
        return model_args.model_name_or_path
    if head_metadata and head_metadata.get("base_model_name_or_path"):
        return head_metadata["base_model_name_or_path"]
    if train_metadata and train_metadata.get("model_name_or_path"):
        return train_metadata["model_name_or_path"]
    raise ValueError(
        "Cannot resolve base encoder path. Please pass --model_name_or_path or ensure checkpoint has metadata "
        f"file `{CLASSIFICATION_HEAD_METADATA_FILENAME}` or `{TRAIN_METADATA_FILENAME}`."
    )


def _load_state_dict_for_eval(checkpoint_dir: str):
    head_path = os.path.join(checkpoint_dir, CLASSIFICATION_HEAD_STATE_FILENAME)
    if os.path.isfile(head_path):
        state_dict = torch.load(head_path, map_location="cpu")
        return state_dict, "classifier_head"

    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
        return state_dict, "full_model_bin"

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file as load_safetensors_file

        state_dict = load_safetensors_file(safetensors_path, device="cpu")
        return state_dict, "full_model_safetensors"

    raise FileNotFoundError(
        f"No checkpoint weights found in {checkpoint_dir}. Expected one of: "
        f"{CLASSIFICATION_HEAD_STATE_FILENAME}, pytorch_model.bin, model.safetensors"
    )


def _normalize_state_dict_keys(state_dict: dict):
    removable_prefixes = ("module.", "_orig_mod.")
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in removable_prefixes:
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        normalized[normalized_key] = value
    return normalized


def _load_tokenizer(checkpoint_dir: str, base_model_name_or_path: str):
    try:
        return _load_clmbr_tokenizer(checkpoint_dir)
    except Exception:
        rank0_print("Tokenizer not found in checkpoint dir; falling back to base model tokenizer.")
        return _load_clmbr_tokenizer(base_model_name_or_path)


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

    try:
        if probs.shape[1] == 2:
            auroc = roc_auc_score(labels, probs[:, 1])
        else:
            auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = 0.5

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
            "idx": list(range(len(labels))) if idx_list is None else list(idx_list),
            "label": labels.tolist(),
            "pred": preds.tolist(),
            "prob": probs.tolist(),
        }
    )
    if id2label is not None:
        raw_df["label_name"] = [id2label.get(int(label), label) for label in labels]
        raw_df["pred_name"] = [id2label.get(int(pred), pred) for pred in preds]

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

    if not os.path.isdir(model_args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {model_args.checkpoint_dir}")

    head_metadata = _load_json_if_exists(os.path.join(model_args.checkpoint_dir, CLASSIFICATION_HEAD_METADATA_FILENAME))
    train_metadata = _load_json_if_exists(os.path.join(model_args.checkpoint_dir, TRAIN_METADATA_FILENAME))
    base_model_name_or_path = _resolve_base_model_path(model_args, head_metadata, train_metadata)

    candidates, label_to_id, id_to_label = _build_label_metadata(data_args.task_name)
    if train_metadata and train_metadata.get("task_name") and train_metadata["task_name"] != data_args.task_name:
        raise ValueError(
            f"Task mismatch: checkpoint was trained for '{train_metadata['task_name']}' "
            f"but evaluation task is '{data_args.task_name}'."
        )
    if head_metadata and head_metadata.get("num_labels") is not None:
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

    tokenizer = _load_tokenizer(model_args.checkpoint_dir, base_model_name_or_path)
    model = LlamaMEDSClassifier(
        model_name_or_path=base_model_name_or_path,
        num_labels=len(candidates),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        freeze_encoder=True,
        tokenizer_vocab_size=int(tokenizer.vocab_size),
    )
    adapter_config_path = os.path.join(model_args.checkpoint_dir, "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        from peft import PeftModel

        model.encoder = PeftModel.from_pretrained(model.encoder, model_args.checkpoint_dir, is_trainable=False)
        rank0_print("Loaded PEFT adapter from checkpoint.")

    state_dict, checkpoint_kind = _load_state_dict_for_eval(model_args.checkpoint_dir)
    state_dict = _normalize_state_dict_keys(state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    missing_keys = list(getattr(load_result, "missing_keys", []))
    missing_classifier_keys = [key for key in missing_keys if key.startswith("classifier.")]
    if unexpected_keys:
        raise ValueError(f"Unexpected keys while loading checkpoint: {unexpected_keys}")
    if missing_classifier_keys:
        raise ValueError(f"Classifier keys missing while loading checkpoint: {missing_classifier_keys}")
    rank0_print(f"Merged checkpoint type: {checkpoint_kind}")

    sample_info_test_path = data_args.sample_info_test_path or os.path.join(
        data_args.processed_dir, "sample_info_test.json"
    )
    if not os.path.isfile(sample_info_test_path):
        raise FileNotFoundError(f"Test sample_info JSON not found: {sample_info_test_path}")

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
    if len(eval_dataset) == 0:
        rank0_print(f"[SKIP] Empty test dataset for task={data_args.task_name}.")
        return

    data_collator = EICUMEDSDataCollator(
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_seq_length=data_args.max_seq_length,
        task_name=data_args.task_name,
    )

    output_dir = data_args.output_dir or os.path.join(model_args.checkpoint_dir, "eval_logs")
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

    num_rows = len(predict_outputs.label_ids) if predict_outputs.label_ids is not None else len(
        predict_outputs.predictions
    )
    idx_list = list(range(num_rows))
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
