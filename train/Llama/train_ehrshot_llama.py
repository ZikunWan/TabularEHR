import json
import os
import sys
import shutil
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info
from utils.metrics import compute_classification_metrics

CLASSIFICATION_HEAD_STATE_FILENAME = "classification_head.bin"
CLASSIFICATION_HEAD_METADATA_FILENAME = "sequence_classification_head_config.json"
TRAIN_METADATA_FILENAME = "sequence_classification_metadata.json"


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print(*args, **kwargs)


def _load_clmbr_tokenizer(model_name_or_path: str):
    from hf_ehr.data.tokenization import CLMBRTokenizer

    # `hf_ehr`'s `from_pretrained()` expects a Hub repo id.
    # For local checkpoints, load tokenizer config directly from disk.
    path = Path(model_name_or_path)
    candidates = []
    if path.is_file():
        candidates.append(path)
    elif path.is_dir():
        candidates.extend(
            [
                path / "tokenizer_config.json",
                path / "tokenizer_config_filtered.json",
            ]
        )

    for config_path in candidates:
        if config_path.is_file():
            # `hf_ehr` tokenizer writes a `versions/` folder next to tokenizer_config.
            # If source dir is read-only (common for shared model weights), copy to writable cache first.
            if os.access(config_path.parent, os.W_OK):
                return CLMBRTokenizer(path_to_tokenizer_config=str(config_path))

            cache_root = Path(os.environ.get("HF_EHR_TOKENIZER_CACHE_DIR", "/tmp/hf_ehr_tokenizers"))
            cache_key = hashlib.sha1(str(config_path.resolve()).encode("utf-8")).hexdigest()[:16]
            cache_dir = cache_root / cache_key
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_config = cache_dir / "tokenizer_config.json"
            if not cached_config.exists():
                shutil.copy2(config_path, cached_config)
            return CLMBRTokenizer(path_to_tokenizer_config=str(cached_config))

    # Fallback to Hub behavior when not a local path.
    return CLMBRTokenizer.from_pretrained(model_name_or_path)


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    task_type = task_info["task_type"]

    if task_type == "binary_classification":
        candidates = ["0", "1"]
    elif task_type == "multi_class_classification":
        candidates = [str(index) for index in range(int(task_info["num_classes"]))]
    else:
        raise ValueError(f"Unsupported EHRShot task_type '{task_type}' for task '{task_name}'.")

    label_to_id = {label: idx for idx, label in enumerate(candidates)}
    id_to_label = {idx: label for idx, label in enumerate(candidates)}
    return candidates, label_to_id, id_to_label


def _extract_classifier_state_dict(state_dict: dict):
    prefixes = ("classifier.", "score.", "classification_head.")
    removable_prefixes = ("module.", "_orig_mod.")
    head_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in removable_prefixes:
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        if normalized_key.startswith(prefixes):
            head_state_dict[normalized_key] = value.detach().cpu()
    if not head_state_dict:
        raise ValueError("No classifier parameters found in state_dict.")
    return head_state_dict


def _save_classifier_head(model, output_dir: str, state_dict: dict):
    os.makedirs(output_dir, exist_ok=True)
    head_state_dict = _extract_classifier_state_dict(state_dict)
    torch.save(head_state_dict, os.path.join(output_dir, CLASSIFICATION_HEAD_STATE_FILENAME))

    metadata = {
        "base_model_name_or_path": model.config._name_or_path,
        "num_labels": int(model.config.num_labels),
        "label2id": model.config.label2id,
        "id2label": {str(key): value for key, value in model.config.id2label.items()},
    }
    with open(os.path.join(output_dir, CLASSIFICATION_HEAD_METADATA_FILENAME), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _copy_tokenizer_config_to_output(tokenizer, output_dir: str):
    tokenizer_config_path = getattr(tokenizer, "path_to_tokenizer_config", None)
    if isinstance(tokenizer_config_path, str) and os.path.isfile(tokenizer_config_path):
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_tokenizer_path = os.path.join(output_dir, "tokenizer_config.json")
        shutil.copy2(tokenizer_config_path, checkpoint_tokenizer_path)
        rank0_print(f"Saved tokenizer config to {checkpoint_tokenizer_path}")


def _save_training_metadata(
    output_dir: str,
    model_args,
    data_args,
    tokenizer,
    *,
    task_name: str,
):
    metadata = {
        "model_name_or_path": model_args.model_name_or_path,
        "tokenizer_config_path": getattr(tokenizer, "path_to_tokenizer_config", None),
        "task_name": task_name,
        "freeze_encoder": bool(model_args.freeze_encoder),
        "use_peft": bool(model_args.use_peft),
        "lora_r": int(model_args.lora_r),
        "lora_alpha": int(model_args.lora_alpha),
        "lora_dropout": float(model_args.lora_dropout),
        "lora_target_modules": model_args.lora_target_modules,
    }
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, TRAIN_METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    rank0_print(f"Saved training metadata to {metadata_path}")


class HeadOnlySequenceClassificationTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        model_to_save = self.accelerator.unwrap_model(self.model)
        _save_classifier_head(model_to_save, output_dir, state_dict=model_to_save.state_dict())
        if getattr(model_to_save, "use_peft", False):
            model_to_save.encoder.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr",
        metadata={"help": "Path to StanfordShahLab llama-base-4096-clmbr weights."},
    )
    tokenizer_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional tokenizer_config.json path (or directory containing it)."},
    )
    freeze_encoder: bool = field(
        default=True,
        metadata={"help": "Freeze encoder parameters and train only classifier head."},
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Enable LoRA adapters for encoder PEFT fine-tuning."},
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated module names for LoRA injection."},
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "Stop training when eval metric does not improve for N evaluation calls."},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum metric improvement required to reset early stopping counter."},
    )


@dataclass
class DataArguments:
    root_dir: str = field(
        default="/data/EHR_data_public/EHRSHOT",
        metadata={"help": "Root directory for EHRSHOT data."},
    )
    train_info_path: str = field(
        default="/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv",
        metadata={"help": "Path to EHRShot training index CSV."},
    )
    val_info_path: str = field(
        default="/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv",
        metadata={"help": "Path to EHRShot validation index CSV."},
    )
    task_name: str = field(
        default="lab_anemia",
        metadata={"help": "Single EHRShot task name."},
    )
    lazy_mode: bool = field(default=True, metadata={"help": "Load EHRSHOT samples lazily."})
    table_mode: str = field(
        default="text_only",
        metadata={"help": "EHRSHOT table mode (unused by MEDS tokenization but kept for compatibility)."},
    )
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Maximum train samples."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Maximum validation samples."})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum tokenized sequence length."})


class MEDSDataCollator:
    def __init__(self, tokenizer, label_to_id: dict, max_seq_length: int, task_name: str):
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_seq_length = max_seq_length
        self.task_name = task_name

    @staticmethod
    def _squeeze_single_batch(tokenized):
        row = {}
        for key, value in tokenized.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                row[key] = value[0]
            else:
                row[key] = value
        return row

    def _tokenize_sample(self, sample: dict):
        events = sample["hf_ehr_events"]

        tokenized = self.tokenizer(
            [events],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None,
        )
        row = self._squeeze_single_batch(tokenized)

        label_text = str(sample["output"])
        if label_text not in self.label_to_id:
            raise ValueError(
                f"Unexpected label '{label_text}' for EHRShot task '{self.task_name}'. "
                f"Expected one of: {sorted(self.label_to_id.keys())}"
            )
        row["labels"] = self.label_to_id[label_text]
        return row

    def __call__(self, features):
        encoded_rows = [self._tokenize_sample(feature) for feature in features]
        labels = torch.tensor([row["labels"] for row in encoded_rows], dtype=torch.long)
        model_features = [{k: v for k, v in row.items() if k != "labels"} for row in encoded_rows]

        batch = self.tokenizer.pad(model_features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


class LlamaMEDSClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        id_to_label: dict,
        label_to_id: dict,
        freeze_encoder: bool = False,
        tokenizer_vocab_size: Optional[int] = None,
        use_peft: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.use_peft = use_peft
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.encoder.config.use_cache = False
        self.num_labels = num_labels

        if tokenizer_vocab_size is not None:
            current_vocab_size = int(self.encoder.get_input_embeddings().num_embeddings)
            if int(tokenizer_vocab_size) != current_vocab_size:
                self.encoder.resize_token_embeddings(int(tokenizer_vocab_size))
                rank0_print(
                    f"Resized token embeddings: {current_vocab_size} -> {int(tokenizer_vocab_size)}"
                )

        if use_peft:
            from peft import LoraConfig, get_peft_model

            target_modules = [module.strip() for module in lora_target_modules.split(",") if module.strip()]
            if not target_modules:
                raise ValueError("--lora_target_modules must contain at least one module name when --use_peft=True.")

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

        hidden_size = int(self.encoder.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_encoder and not use_peft:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.config = self.encoder.config
        self.config.num_labels = num_labels
        self.config.id2label = id_to_label
        self.config.label2id = label_to_id
        self.config.problem_type = "single_label_classification"

    @staticmethod
    def _pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        seq_lens = attention_mask.long().sum(dim=1) - 1
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, seq_lens, :]

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # When encoder is wrapped by PEFT (e.g., LoRA), call the wrapper forward so
        # adapter weights participate in computation; otherwise keep the lightweight
        # direct backbone path.
        if hasattr(self.encoder, "peft_config"):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
                **kwargs,
            )
            hidden_states = outputs.hidden_states[-1]
        else:
            outputs = self.encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state

        pooled = self._pool_last_token(hidden_states, attention_mask)
        # Keep classifier matmul dtype-consistent when encoder runs in bf16
        # while the head remains fp32 (common with DeepSpeed + non-autocast paths).
        pooled = pooled.to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


def _load_source_dataset(data_args, sample_info_path: str, split_name: str, max_samples: Optional[int]):
    dataset = EHRSHOTDataset(
        root_dir=data_args.root_dir,
        sample_info_path=sample_info_path,
        task_name=data_args.task_name,
        lazy_mode=data_args.lazy_mode,
        table_mode=data_args.table_mode,
        max_samples=max_samples,
        return_meds=True,
    )
    rank0_print(f"{split_name} source [{data_args.task_name}, MEDS] size: {len(dataset)}")
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.task_name not in get_task_info():
        raise ValueError(f"Unsupported EHRShot task_name '{data_args.task_name}'.")

    train_info_path = data_args.train_info_path
    val_info_path = data_args.val_info_path
    if not os.path.isfile(train_info_path):
        raise FileNotFoundError(f"Training index CSV not found: {train_info_path}")

    training_args.remove_unused_columns = False
    training_args.save_safetensors = True
    training_args.seed = getattr(training_args, "seed", 42) or 42
    training_args.bf16 = True
    training_args.fp16 = False
    set_seed(training_args.seed)

    if training_args.eval_strategy != "no" and not os.path.isfile(val_info_path):
        raise FileNotFoundError(f"Validation index CSV not found: {val_info_path}")

    if (not model_args.freeze_encoder) and (not model_args.use_peft):
        raise ValueError(
            "Full encoder fine-tuning is disabled for this script. "
            "Use --use_peft True when setting --freeze_encoder False."
        )
    if model_args.early_stopping_patience < 1:
        raise ValueError("--early_stopping_patience must be >= 1.")

    if training_args.eval_strategy == "no":
        raise ValueError("Early stopping requires evaluation. Please set --eval_strategy to 'steps' or 'epoch'.")
    # Early stopping / best-checkpoint selection should follow AUROC for classification.
    training_args.metric_for_best_model = "eval_auroc"
    training_args.greater_is_better = True
    training_args.load_best_model_at_end = True

    candidates, label_to_id, id_to_label = _build_label_metadata(data_args.task_name)
    tokenizer_source = model_args.tokenizer_config_path or model_args.model_name_or_path
    tokenizer = _load_clmbr_tokenizer(tokenizer_source)

    rank0_print("=" * 80)
    rank0_print("EHRSHOT MEDS Llama Encoder Classifier Train")
    rank0_print("=" * 80)
    rank0_print(f"Model path: {model_args.model_name_or_path}")
    rank0_print(f"Tokenizer source: {tokenizer_source}")
    rank0_print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    rank0_print(f"Task: {data_args.task_name}")
    rank0_print(f"Train index: {train_info_path}")
    rank0_print(f"Validation index: {val_info_path}")
    rank0_print(f"Max seq length: {data_args.max_seq_length}")
    rank0_print(f"Freeze encoder: {model_args.freeze_encoder}")
    rank0_print(f"Use PEFT: {model_args.use_peft}")
    if model_args.use_peft:
        rank0_print(
            f"LoRA config: r={model_args.lora_r}, alpha={model_args.lora_alpha}, "
            f"dropout={model_args.lora_dropout}, target_modules={model_args.lora_target_modules}"
        )
    rank0_print(f"Early stopping patience: {model_args.early_stopping_patience}")
    rank0_print(f"Early stopping threshold: {model_args.early_stopping_threshold}")
    rank0_print(f"Metric for best model: {training_args.metric_for_best_model}")
    rank0_print(f"Output dir: {training_args.output_dir}")

    model = LlamaMEDSClassifier(
        model_name_or_path=model_args.model_name_or_path,
        num_labels=len(candidates),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        freeze_encoder=model_args.freeze_encoder,
        tokenizer_vocab_size=int(tokenizer.vocab_size),
        use_peft=model_args.use_peft,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        lora_target_modules=model_args.lora_target_modules,
    )

    trainable_parameters = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if model_args.use_peft:
        encoder_trainable = [name for name in trainable_parameters if name.startswith("encoder.")]
        if not encoder_trainable:
            raise ValueError("PEFT mode expects trainable encoder adapter parameters, but none were found.")
    elif model_args.freeze_encoder:
        invalid_trainable = [name for name in trainable_parameters if not name.startswith("classifier.")]
        if invalid_trainable:
            raise ValueError(
                "Head-only training requires only classifier parameters to be trainable, "
                f"but found non-classifier trainable parameters: {invalid_trainable[:5]}"
            )
    rank0_print(f"Trainable parameter tensors: {len(trainable_parameters)}")
    rank0_print(
        f"Trainable parameter names: {', '.join(trainable_parameters) if trainable_parameters else 'None'}"
    )

    train_dataset = _load_source_dataset(
        data_args=data_args,
        sample_info_path=train_info_path,
        split_name="train",
        max_samples=data_args.max_train_samples,
    )
    rank0_print(f"Final train dataset size: {len(train_dataset)}")

    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = _load_source_dataset(
            data_args=data_args,
            sample_info_path=val_info_path,
            split_name="validation",
            max_samples=data_args.max_eval_samples,
        )
        rank0_print(f"Final validation dataset size: {len(eval_dataset)}")

    data_collator = MEDSDataCollator(
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_seq_length=data_args.max_seq_length,
        task_name=data_args.task_name,
    )

    trainer = HeadOnlySequenceClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_classification_metrics if eval_dataset is not None else None,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping_patience,
                early_stopping_threshold=model_args.early_stopping_threshold,
            )
        ],
    )

    rank0_print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    rank0_print(f"Saving checkpoint to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    _copy_tokenizer_config_to_output(tokenizer, training_args.output_dir)
    _save_training_metadata(
        output_dir=training_args.output_dir,
        model_args=model_args,
        data_args=data_args,
        tokenizer=tokenizer,
        task_name=data_args.task_name,
    )

if __name__ == "__main__":
    main()
