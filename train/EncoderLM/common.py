import json
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

IGNORE_INDEX = 0


@dataclass
class LocalEncoderLMScriptArguments:
    max_seq_length: Optional[int] = field(default=2048)
    system_prompt: str = field(default="")


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print(*args, **kwargs)


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


def freeze_for_head_only_sequence_classification(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    enabled = False
    for attr_name in ("classifier", "score", "classification_head"):
        module = getattr(model, attr_name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True
            enabled = True

    if not enabled:
        for name, parameter in model.named_parameters():
            if any(key in name for key in ("classifier.", "score.", "classification_head.")):
                parameter.requires_grad = True
                enabled = True

    if not enabled:
        raise ValueError(
            f"Unable to locate a classification head on model class {model.__class__.__name__} for head-only training."
        )

    return model


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

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


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
