import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, IterableDataset
from peft import LoraConfig, TaskType
from transformers import Trainer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
)

IGNORE_INDEX = 0
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
CLASSIFICATION_HEAD_STATE_FILENAME = "classification_head.bin"
CLASSIFICATION_HEAD_METADATA_FILENAME = "sequence_classification_head_config.json"


@dataclass
class LocalLLMScriptArguments:
    max_seq_length: Optional[int] = field(default=8192)
    system_prompt: str = field(default="You are a helpful medical AI assistant specialized in EHR analysis.")
    use_sequence_classification: bool = field(default=False)


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print(*args, **kwargs)


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


def extract_classification_head_state_dict(state_dict: dict):
    prefixes = ("classifier.", "score.", "classification_head.")
    head_state_dict = {
        key: value.detach().cpu()
        for key, value in state_dict.items()
        if key.startswith(prefixes)
    }
    if not head_state_dict:
        raise ValueError("No classification head parameters were found in the state_dict.")
    return head_state_dict


def save_sequence_classification_head(model, output_dir: str, state_dict: Optional[dict] = None):
    os.makedirs(output_dir, exist_ok=True)
    if state_dict is None:
        state_dict = model.state_dict()
    head_state_dict = extract_classification_head_state_dict(state_dict)
    torch.save(head_state_dict, os.path.join(output_dir, CLASSIFICATION_HEAD_STATE_FILENAME))

    metadata = {
        "base_model_name_or_path": getattr(model.config, "_name_or_path", None),
        "num_labels": int(model.config.num_labels),
        "label2id": getattr(model.config, "label2id", None),
        "id2label": {str(key): value for key, value in getattr(model.config, "id2label", {}).items()},
    }
    with open(os.path.join(output_dir, CLASSIFICATION_HEAD_METADATA_FILENAME), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


class HeadOnlySequenceClassificationTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        model_to_save = self.accelerator.unwrap_model(self.model)
        save_sequence_classification_head(model_to_save, output_dir, state_dict=state_dict)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


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
            rank0_print("gpt-oss flash_attention_2 is unavailable in this environment; falling back to eager attention.")
            return model_cls.from_pretrained(
                model_name_or_path,
                **fallback_kwargs,
            )
        raise


def load_model(
    model_name_or_path: str,
    use_sequence_classification: bool = False,
    num_labels: Optional[int] = None,
    label2id: Optional[dict] = None,
    id2label: Optional[dict] = None,
):
    family = detect_model_family(model_name_or_path)
    attn_implementation = "eager" if family == "gpt_oss" else "sdpa"

    if use_sequence_classification:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        _set_sequence_classification_metadata(config, num_labels, label2id, id2label, family)
        model_kwargs = {
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
            "attn_implementation": attn_implementation,
            "config": config,
            "ignore_mismatched_sizes": num_labels is not None,
        }

        model = _load_model_with_fallback(
            AutoModelForSequenceClassification,
            model_name_or_path,
            model_kwargs,
            family,
        )
        model.config.use_cache = False
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            model.config.text_config.use_cache = False
        if family == "medgemma_1_5":
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
            tokenizer = processor.tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            tokenizer.padding_side = "right"
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = tokenizer.pad_token_id
            return model, processor

        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": True,
        }
        if family == "ehr_r1":
            tokenizer_kwargs["use_fast"] = False
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "right"
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    if family == "medgemma_1_5":
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        tokenizer = processor.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        tokenizer.padding_side = "right"
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model, processor

    model_kwargs = {
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }
    model = _load_model_with_fallback(
        AutoModelForCausalLM,
        model_name_or_path,
        model_kwargs,
        family,
    )
    tokenizer_kwargs = {
        "trust_remote_code": True,
        "use_fast": True,
    }
    if family == "ehr_r1":
        tokenizer_kwargs["use_fast"] = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
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
    user_text = "\n".join(part for part in [input_text.strip(), instruction.strip()] if part)
    family = detect_model_family(model_name_or_path)

    if family == "medgemma_1_5":
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_text}]},
        ]
        if output_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": output_text}]})
        return messages

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
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
            add_generation_prompt=output_text is None,
        )

    chat_template_kwargs = {}
    if family in {"qwen3_5", "ehr_r1"}:
        chat_template_kwargs["enable_thinking"] = False

    return processor_or_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=output_text is None,
        **chat_template_kwargs,
    )


def tokenize_and_truncate(
    processor_or_tokenizer,
    text: str,
    max_seq_length: Optional[int] = None,
    completion_only_loss: bool = False,
    prompt_text: Optional[str] = None,
):
    tokenizer = processor_or_tokenizer.tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else processor_or_tokenizer
    model_name_or_path = tokenizer.name_or_path
    family = detect_model_family(model_name_or_path)

    if completion_only_loss:
        if prompt_text is None:
            raise ValueError("prompt_text is required when completion_only_loss=True")

        tokenized = tokenizer(text)
        if family == "medgemma_1_5" and "token_type_ids" not in tokenized:
            tokenized["token_type_ids"] = [0] * len(tokenized["input_ids"])
        prompt_token_ids = tokenizer(prompt_text)["input_ids"]
        prompt_token_len = len(prompt_token_ids)
        tokenized["completion_mask"] = [IGNORE_INDEX] * prompt_token_len + tokenized["input_ids"][prompt_token_len:]

        if max_seq_length is not None and max_seq_length > 0:
            tokenized = {k: v[-max_seq_length:] for k, v in tokenized.items()}
        return tokenized

    tokenized = tokenizer(text, return_tensors="pt")
    if family == "medgemma_1_5" and "token_type_ids" not in tokenized:
        tokenized["token_type_ids"] = torch.zeros_like(tokenized["input_ids"])
    if max_seq_length is not None and max_seq_length > 0:
        tokenized = {k: v[:, -max_seq_length:] for k, v in tokenized.items()}
    return tokenized


def prepare_training_args(training_args):
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    if getattr(training_args, "completion_only_loss", False):
        training_args.use_liger_kernel = False
    return training_args


def build_lora_config(model_config):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=getattr(model_config, "lora_r", 16),
        lora_alpha=getattr(model_config, "lora_alpha", 32),
        lora_dropout=getattr(model_config, "lora_dropout", 0.05),
        target_modules=getattr(model_config, "lora_target_modules", None) or DEFAULT_LORA_TARGET_MODULES,
        modules_to_save=getattr(model_config, "lora_modules_to_save", None),
        use_rslora=getattr(model_config, "use_rslora", False),
        use_dora=getattr(model_config, "use_dora", False),
    )


def build_dataset_from_source(
    base_dataset,
    processor_or_tokenizer,
    max_seq_length: Optional[int],
    system_prompt: str,
    shuffle: bool = False,
    iterable: bool = False,
    completion_only_loss: bool = False,
):
    if hasattr(processor_or_tokenizer, "tokenizer"):
        model_name_or_path = processor_or_tokenizer.tokenizer.name_or_path
    else:
        model_name_or_path = processor_or_tokenizer.name_or_path

    def _format_sample(sample):
        input_text = str(sample.get("input", ""))
        instruction = str(sample.get("instruction", ""))
        output_text = str(sample.get("output", ""))

        full_text = apply_template(
            model_name_or_path=model_name_or_path,
            processor_or_tokenizer=processor_or_tokenizer,
            input_text=input_text,
            instruction=instruction,
            system_prompt=system_prompt,
            output_text=output_text,
        )

        prompt_text = None
        if completion_only_loss:
            prompt_text = apply_template(
                model_name_or_path=model_name_or_path,
                processor_or_tokenizer=processor_or_tokenizer,
                input_text=input_text,
                instruction=instruction,
                system_prompt=system_prompt,
                output_text=None,
            )

        tokenized = tokenize_and_truncate(
            processor_or_tokenizer=processor_or_tokenizer,
            text=full_text,
            max_seq_length=max_seq_length,
            completion_only_loss=completion_only_loss,
            prompt_text=prompt_text,
        )

        formatted = {}
        for key, value in tokenized.items():
            if torch.is_tensor(value):
                formatted[key] = value[0].tolist() if value.ndim > 1 else value.tolist()
            else:
                formatted[key] = value
        return formatted

    def _generator():
        for sample in base_dataset:
            yield _format_sample(sample)

    if iterable:
        return IterableDataset.from_generator(_generator)

    dataset = Dataset.from_list(list(_generator()))
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    return dataset
