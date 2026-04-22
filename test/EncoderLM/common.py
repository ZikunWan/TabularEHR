import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


IGNORE_INDEX = 0


def _infer_model_type_from_config(model_name_or_path: str) -> Optional[str]:
    config_path = os.path.join(model_name_or_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("model_type")
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        return getattr(config, "model_type", None)
    except Exception:
        return None


def detect_model_family(model_name_or_path: str) -> str:
    normalized = (model_name_or_path or "").strip().lower()
    if "gatortron" in normalized:
        return "gatortron"
    model_type = _infer_model_type_from_config(model_name_or_path)
    if model_type == "megatron-bert":
        return "gatortron"
    raise ValueError("Unsupported encoder model family. Explicit support is only implemented for GatorTron.")


def load_tokenizer(model_name_or_path: str, local_files_only: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_name_or_path: str,
    num_labels: Optional[int] = None,
    label2id: Optional[dict] = None,
    id2label: Optional[dict] = None,
):
    detect_model_family(model_name_or_path)

    model_kwargs = {
        "trust_remote_code": True,
    }
    if num_labels is not None:
        model_kwargs["num_labels"] = num_labels
    if label2id is not None:
        model_kwargs["label2id"] = label2id
    if id2label is not None:
        model_kwargs["id2label"] = id2label

    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )


def build_messages(
    model_name_or_path: str,
    input_text: str,
    instruction: str,
    system_prompt: str,
    output_text: Optional[str] = None,
):
    detect_model_family(model_name_or_path)
    user_text = "\n".join(part for part in [input_text.strip(), instruction.strip()] if part)
    messages = [user_text]
    if output_text is not None:
        messages.append(output_text)
    return messages


def apply_template(
    model_name_or_path: str,
    processor_or_tokenizer,
    input_text: str,
    instruction: str,
    system_prompt: str,
    output_text: Optional[str] = None,
):
    messages = build_messages(
        model_name_or_path=model_name_or_path,
        input_text=input_text,
        instruction=instruction,
        system_prompt=system_prompt,
        output_text=output_text,
    )
    return "\n\n".join(messages)


def normalize_label(raw_label) -> str:
    label = str(raw_label).strip()
    while len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
        label = label[1:-1].strip()

    lowered = label.lower()
    if lowered in {"yes", "y", "true"}:
        return "yes"
    if lowered in {"no", "n", "false"}:
        return "no"
    return lowered


def augment_binary_aliases(label_to_id):
    aliases = {}
    if "0" in label_to_id and "1" in label_to_id:
        aliases.update(
            {
                "no": label_to_id["0"],
                "false": label_to_id["0"],
                "n": label_to_id["0"],
                "yes": label_to_id["1"],
                "true": label_to_id["1"],
                "y": label_to_id["1"],
            }
        )
    if "no" in label_to_id and "yes" in label_to_id:
        aliases.update(
            {
                "0": label_to_id["no"],
                "false": label_to_id["no"],
                "n": label_to_id["no"],
                "1": label_to_id["yes"],
                "true": label_to_id["yes"],
                "y": label_to_id["yes"],
            }
        )
    label_to_id.update(aliases)


def resolve_label_id(raw_label, label_to_id, label_normalizer=None):
    label = label_normalizer(raw_label) if label_normalizer is not None else str(raw_label)
    if label not in label_to_id:
        raise KeyError(
            f"Unknown label '{raw_label}' (normalized='{label}'). "
            f"Supported labels: {sorted(label_to_id.keys())}"
        )
    return label_to_id[label]


def build_prompts_from_dataset(
    dataset,
    processor_or_tokenizer,
    system_prompt: str = "",
    max_seq_length: Optional[int] = None,
):
    tokenizer = processor_or_tokenizer.tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else processor_or_tokenizer
    model_name_or_path = tokenizer.name_or_path

    prompts = []
    meta_list = []
    for index in range(len(dataset)):
        sample = dataset[index]
        instruction = sample.get("instruction")
        if instruction is None:
            instruction = sample.get("task_info", {}).get("instruction")
        if instruction is None or str(instruction).strip() == "":
            raise KeyError(
                f"Missing non-empty instruction for sample idx={index}. "
                "Expected `sample['instruction']` or `sample['task_info']['instruction']`."
            )
        prompt = apply_template(
            model_name_or_path=model_name_or_path,
            processor_or_tokenizer=processor_or_tokenizer,
            input_text=str(sample.get("input", "")),
            instruction=str(instruction),
            system_prompt=system_prompt,
            output_text=None,
        )
        if max_seq_length is not None and max_seq_length > 0:
            prompt_token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if len(prompt_token_ids) > max_seq_length:
                prompt = tokenizer.decode(
                    prompt_token_ids[-max_seq_length:],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
        prompts.append(prompt)
        meta_list.append(
            {
                "label": sample.get("output", sample.get("task_info", {}).get("label")),
                "idx": index,
            }
        )

    return prompts, meta_list


def tokenize_classification_prompts(
    prompts,
    meta_list,
    tokenizer,
    label_to_id,
    max_seq_len: int,
    label_normalizer=None,
):
    from datasets import Dataset

    rows = []
    for prompt, meta in zip(prompts, meta_list):
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_len,
            return_token_type_ids=True,
        )
        tokenized["labels"] = resolve_label_id(meta["label"], label_to_id, label_normalizer=label_normalizer)
        tokenized["idx"] = meta["idx"]
        rows.append(tokenized)
    return Dataset.from_list(rows)


def compute_sequence_classification_metrics(
    logits,
    labels,
    task_name: str,
    idx_list=None,
    id2label: Optional[dict] = None,
):
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


def tokenize_and_truncate(
    processor_or_tokenizer,
    text: str,
    max_seq_length: Optional[int] = None,
    completion_only_loss: bool = False,
    prompt_text: Optional[str] = None,
):
    tokenizer = processor_or_tokenizer.tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else processor_or_tokenizer

    if completion_only_loss:
        if prompt_text is None:
            raise ValueError("prompt_text is required when completion_only_loss=True")

        tokenized = tokenizer(text)
        prompt_token_ids = tokenizer(prompt_text)["input_ids"]
        prompt_token_len = len(prompt_token_ids)
        tokenized["completion_mask"] = [IGNORE_INDEX] * prompt_token_len + tokenized["input_ids"][prompt_token_len:]

        if max_seq_length is not None and max_seq_length > 0:
            tokenized = {k: v[-max_seq_length:] for k, v in tokenized.items()}
        return tokenized

    tokenized = tokenizer(text, return_tensors="pt")
    if max_seq_length is not None and max_seq_length > 0:
        tokenized = {k: v[:, -max_seq_length:] for k, v in tokenized.items()}
    return tokenized
