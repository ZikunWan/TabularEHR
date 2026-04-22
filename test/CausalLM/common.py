import json
import os
import shutil
import sys
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file as save_safetensors_file
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
)
from vllm.lora.request import LoRARequest


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


IGNORE_INDEX = 0
CLASSIFICATION_HEAD_STATE_FILENAME = "classification_head.bin"
CLASSIFICATION_HEAD_METADATA_FILENAME = "sequence_classification_head_config.json"
VLLM_CLASSIFICATION_HEAD_FILENAME = "classification_head.safetensors"


def detect_model_family(model_name_or_path: str) -> str:
    normalized = (model_name_or_path or "").strip().lower()
    if "medgemma-1.5" in normalized:
        return "medgemma_1_5"
    if "medgemma_1_5" in normalized:
        return "medgemma_1_5"
    if "gpt-oss" in normalized:
        return "gpt_oss"
    if "gpt_oss" in normalized:
        return "gpt_oss"
    if "ehr-r1" in normalized:
        return "ehr_r1"
    if "ehr_r1" in normalized:
        return "ehr_r1"
    if "qwen3.5" in normalized:
        return "qwen3_5"
    if "qwen3_5" in normalized:
        return "qwen3_5"
    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model_type = getattr(config, "model_type", None)
        if model_type == "gpt_oss":
            return "gpt_oss"
        if model_type == "qwen3_5":
            return "qwen3_5"
        if model_type == "gemma3":
            return "medgemma_1_5"
        if model_type == "qwen3" and ("ehr" in normalized or "r1" in normalized):
            return "ehr_r1"
    except Exception:
        pass
    raise ValueError(
        "Unsupported model family. Explicit support is only implemented for "
        "MedGemma-1.5, gpt-oss, Qwen3.5, and EHR-R1."
    )


def resolve_tensor_parallel_size(requested_tp_size: int) -> int:
    if requested_tp_size > 0:
        return requested_tp_size

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        device_count = len([item for item in visible_devices.split(",") if item.strip()])
        if device_count > 0:
            return device_count

    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def load_tokenizer(
    model_name_or_path: str,
    local_files_only: bool = True,
    use_sequence_classification: bool = False,
):
    family = detect_model_family(model_name_or_path)
    if family == "medgemma_1_5":
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "left"
        return processor

    tokenizer_kwargs = {
        "trust_remote_code": True,
        "use_fast": True,
        "local_files_only": local_files_only,
    }
    if family == "ehr_r1":
        tokenizer_kwargs["use_fast"] = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_model_with_fallback(model_cls, model_name_or_path: str, model_kwargs: dict, family: str):
    try:
        return model_cls.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
    except ImportError as error:
        if family == "gpt_oss" and model_kwargs.get("attn_implementation") == "flash_attention_2" and "kernels" in str(error):
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["attn_implementation"] = "eager"
            print("gpt-oss flash_attention_2 is unavailable in this environment; falling back to eager attention.")
            return model_cls.from_pretrained(
                model_name_or_path,
                **fallback_kwargs,
            )
        raise


def _set_sequence_classification_metadata(config, num_labels, label2id, id2label, family: str):
    if num_labels is not None:
        config.num_labels = num_labels
    if label2id is not None:
        config.label2id = label2id
    if id2label is not None:
        config.id2label = id2label
    config.problem_type = None

    if family in {"qwen3_5", "medgemma_1_5"} and hasattr(config, "text_config") and config.text_config is not None:
        text_config = config.text_config
        if num_labels is not None:
            text_config.num_labels = num_labels
        if label2id is not None:
            text_config.label2id = label2id
        if id2label is not None:
            text_config.id2label = id2label
        text_config.problem_type = None


def load_model(
    model_name_or_path: str,
    use_sequence_classification: bool = False,
    num_labels: Optional[int] = None,
    label2id: Optional[dict] = None,
    id2label: Optional[dict] = None,
    local_files_only: bool = True,
):
    if not use_sequence_classification:
        raise ValueError("load_model in test/CausalLM/common.py is only available when use_sequence_classification=True.")

    metadata_path = os.path.join(model_name_or_path, CLASSIFICATION_HEAD_METADATA_FILENAME)
    head_state_path = os.path.join(model_name_or_path, CLASSIFICATION_HEAD_STATE_FILENAME)
    metadata = None
    source_model_path = model_name_or_path
    if os.path.isfile(metadata_path) and os.path.isfile(head_state_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        source_model_path = metadata["base_model_name_or_path"]
        if num_labels is None:
            num_labels = metadata.get("num_labels")
        if label2id is None:
            label2id = metadata.get("label2id")
        if id2label is None and metadata.get("id2label") is not None:
            id2label = {int(key): value for key, value in metadata["id2label"].items()}

    family = detect_model_family(source_model_path)
    attn_implementation = "eager" if family == "gpt_oss" else "sdpa"
    config = AutoConfig.from_pretrained(
        source_model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    _set_sequence_classification_metadata(config, num_labels, label2id, id2label, family)
    model_kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "attn_implementation": attn_implementation,
        "config": config,
        "ignore_mismatched_sizes": num_labels is not None,
    }
    if torch.cuda.is_available():
        model_kwargs["dtype"] = torch.bfloat16

    model = _load_model_with_fallback(
        AutoModelForSequenceClassification,
        source_model_path,
        model_kwargs,
        family,
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.use_cache = False

    if metadata is not None:
        head_state_dict = torch.load(head_state_path, map_location="cpu")
        load_result = model.load_state_dict(head_state_dict, strict=False)
        unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
        if unexpected_keys:
            raise ValueError(f"Unexpected keys found while loading classification head: {unexpected_keys}")

    tokenizer_source = model_name_or_path if os.path.isfile(os.path.join(model_name_or_path, "tokenizer_config.json")) else source_model_path
    tokenizer_or_processor = load_tokenizer(
        tokenizer_source,
        local_files_only=local_files_only,
        use_sequence_classification=use_sequence_classification,
    )
    tokenizer = tokenizer_or_processor.tokenizer if hasattr(tokenizer_or_processor, "tokenizer") else tokenizer_or_processor
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None and getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = pad_token_id
    if (
        hasattr(model.config, "text_config")
        and model.config.text_config is not None
        and pad_token_id is not None
        and getattr(model.config.text_config, "pad_token_id", None) is None
    ):
        model.config.text_config.pad_token_id = pad_token_id
    return model, tokenizer_or_processor


def materialize_vllm_sequence_classification_model(
    model_name_or_path: str,
    num_labels: Optional[int] = None,
    label2id: Optional[dict] = None,
    id2label: Optional[dict] = None,
    local_files_only: bool = True,
):
    metadata_path = os.path.join(model_name_or_path, CLASSIFICATION_HEAD_METADATA_FILENAME)
    head_state_path = os.path.join(model_name_or_path, CLASSIFICATION_HEAD_STATE_FILENAME)
    metadata = None
    source_model_path = model_name_or_path
    if os.path.isfile(metadata_path) and os.path.isfile(head_state_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        source_model_path = metadata["base_model_name_or_path"]
        if num_labels is None:
            num_labels = metadata.get("num_labels")
        if label2id is None:
            label2id = metadata.get("label2id")
        if id2label is None and metadata.get("id2label") is not None:
            id2label = {int(key): value for key, value in metadata["id2label"].items()}
    else:
        return model_name_or_path, model_name_or_path, False

    family = detect_model_family(source_model_path)
    config = AutoConfig.from_pretrained(
        source_model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    _set_sequence_classification_metadata(config, num_labels, label2id, id2label, family)

    temp_dir = tempfile.mkdtemp(prefix="vllm_seqcls_", dir="/tmp")

    for entry in os.listdir(source_model_path):
        src = os.path.join(source_model_path, entry)
        dst = os.path.join(temp_dir, entry)
        if os.path.isdir(src):
            continue
        if entry in {"config.json", "model.safetensors.index.json"}:
            # These two files are rewritten for the temporary materialized model
            # and must never remain symlinks to the base model directory.
            continue
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    overlay_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "processor_config.json",
        "preprocessor_config.json",
        "added_tokens.json",
    ]
    for entry in overlay_files:
        src = os.path.join(model_name_or_path, entry)
        dst = os.path.join(temp_dir, entry)
        if not os.path.isfile(src):
            continue
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(src, dst)

    config_path = os.path.join(temp_dir, "config.json")
    if os.path.lexists(config_path):
        os.remove(config_path)
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string(use_diff=False))

    head_state_dict = torch.load(head_state_path, map_location="cpu")
    safetensors_state = {
        key: value.detach().cpu().contiguous()
        for key, value in head_state_dict.items()
        if torch.is_tensor(value)
    }
    head_safetensors_path = os.path.join(temp_dir, VLLM_CLASSIFICATION_HEAD_FILENAME)
    save_safetensors_file(safetensors_state, head_safetensors_path)

    index_filename = "model.safetensors.index.json"
    source_index_path = os.path.join(source_model_path, index_filename)
    if os.path.isfile(source_index_path):
        with open(source_index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.setdefault("weight_map", {})
        for key in safetensors_state:
            weight_map[key] = VLLM_CLASSIFICATION_HEAD_FILENAME
        metadata_obj = index_data.setdefault("metadata", {})
        total_size = metadata_obj.get("total_size")
        if isinstance(total_size, int):
            metadata_obj["total_size"] = total_size + sum(
                tensor.numel() * tensor.element_size() for tensor in safetensors_state.values()
            )
        temp_index_path = os.path.join(temp_dir, index_filename)
        if os.path.lexists(temp_index_path):
            os.remove(temp_index_path)
        with open(temp_index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=True, indent=2)

    return temp_dir, temp_dir, True


def cleanup_materialized_vllm_model(model_path: str, was_materialized: bool):
    if was_materialized and os.path.isdir(model_path):
        shutil.rmtree(model_path, ignore_errors=True)


def resolve_vllm_model_and_lora(
    model_path: str,
    base_model_path: Optional[str] = None,
):
    if not base_model_path:
        return model_path, model_path, None

    lora_request = LoRARequest("adapter", 1, model_path)
    return base_model_path, base_model_path, lora_request


def build_messages(
    model_name_or_path: str,
    input_text: str,
    instruction: str,
    system_prompt: str,
    output_text: Optional[str] = None,
):
    user_text = "\n".join(part for part in [input_text.strip(), instruction.strip()] if part)
    family = detect_model_family(model_name_or_path)

    if family == "medgemma_1_5":
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
        if output_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
        return messages

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_text})
    if output_text is not None:
        messages.append({"role": "assistant", "content": output_text})
    return messages


def apply_template(
    model_name_or_path: str,
    processor_or_tokenizer,
    input_text: str,
    instruction: str,
    system_prompt: str,
    output_text: Optional[str] = None,
):
    family = detect_model_family(model_name_or_path)
    messages = build_messages(
        model_name_or_path=model_name_or_path,
        input_text=input_text,
        instruction=instruction,
        system_prompt=system_prompt,
        output_text=output_text,
    )

    if family == "medgemma_1_5":
        return processor_or_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    chat_template_kwargs = {}
    if family in {"qwen3_5", "ehr_r1"}:
        chat_template_kwargs["enable_thinking"] = False

    return processor_or_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )


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
        prompt = apply_template(
            model_name_or_path=model_name_or_path,
            processor_or_tokenizer=processor_or_tokenizer,
            input_text=str(sample.get("input", "")),
            instruction=sample["instruction"],
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
    f1 = f1_score(labels, preds, average="macro")

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
                "Macro F1": float(f1),
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
