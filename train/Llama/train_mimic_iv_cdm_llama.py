import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info
from train.Llama.train_ehrshot_llama import (
    HeadOnlySequenceClassificationTrainer,
    LlamaMEDSClassifier,
    _copy_tokenizer_config_to_output,
    _load_clmbr_tokenizer,
    _save_training_metadata,
    rank0_print,
)


MAIN_DIAGNOSIS_TASK = "MIMIC-IV-CDM Main Disease Diagnoses"


def _normalize_label(raw_label) -> str:
    label = str(raw_label).strip()
    while len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
        label = label[1:-1].strip()
    return label.lower()


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    task_type = task_info["task_type"]

    if task_type != "multi_class_classification":
        raise NotImplementedError(
            f"MEDS encoder training currently supports only multi-class MIMIC-IV-CDM tasks. "
            f"Got task '{task_name}' with task_type='{task_type}'."
        )

    if "candidate" not in task_info:
        raise ValueError(f"Task '{task_name}' is missing `candidate` labels in task_info.")

    candidates = [_normalize_label(candidate) for candidate in task_info["candidate"]]
    label_to_id = {label: idx for idx, label in enumerate(candidates)}
    id_to_label = {idx: label for idx, label in enumerate(candidates)}
    return candidates, label_to_id, id_to_label


class MIMICIVCDMMEDSDataCollator:
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

        raw_label = sample["output"]
        label_text = _normalize_label(raw_label)
        if label_text not in self.label_to_id:
            raise ValueError(
                f"Unexpected label '{raw_label}' (normalized: '{label_text}') for task '{self.task_name}'. "
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


@dataclass
class ModelArguments:
    tokenizer_config_path: str = field(
        metadata={"help": "Path to tokenizer_config.json or directory containing it."},
    )
    model_name_or_path: str = field(
        default="/data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr",
        metadata={"help": "Path to StanfordShahLab llama-base-4096-clmbr weights."},
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


@dataclass
class DataArguments:
    root_dir: str = field(
        default="/data/EHR_data_public/mimic-iv-cdm",
        metadata={"help": "Root directory for MIMIC-IV-CDM data."},
    )
    task_name: str = field(
        default=MAIN_DIAGNOSIS_TASK,
        metadata={"help": f"MIMIC-IV-CDM task name. Recommended: {MAIN_DIAGNOSIS_TASK}"},
    )
    lazy_mode: bool = field(default=True, metadata={"help": "Load MIMIC-IV-CDM samples lazily."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Maximum train samples."})
    max_seq_length: int = field(default=4096, metadata={"help": "Maximum tokenized sequence length."})
    concept_map_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional concept map directory for mapping ITEMID to standard vocab code."},
    )


def _load_source_dataset(data_args, split_name: str, shuffle: bool, max_samples: Optional[int]):
    dataset = MIMICIVCDM(
        root_dir=data_args.root_dir,
        split=split_name,
        lazy_mode=data_args.lazy_mode,
        shuffle=shuffle,
        task_name=data_args.task_name,
        max_samples=max_samples,
        return_meds=True,
        concept_map_dir=data_args.concept_map_dir,
    )
    rank0_print(f"{split_name} source [{data_args.task_name}, MEDS] size: {len(dataset)}")
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    task_info = get_task_info()
    if data_args.task_name not in task_info:
        raise ValueError(f"Unsupported task_name '{data_args.task_name}'. Available tasks: {list(task_info.keys())}")
    if data_args.task_name != MAIN_DIAGNOSIS_TASK:
        raise NotImplementedError(
            "MEDS encoder training currently supports only "
            f"'{MAIN_DIAGNOSIS_TASK}', got '{data_args.task_name}'."
        )

    training_args.remove_unused_columns = False
    training_args.save_safetensors = True
    training_args.bf16 = True
    training_args.fp16 = False
    set_seed(training_args.seed)

    if (not model_args.freeze_encoder) and (not model_args.use_peft):
        raise ValueError(
            "Full encoder fine-tuning is disabled for this script. "
            "Use --use_peft True when setting --freeze_encoder False."
        )
    if training_args.eval_strategy != "no":
        rank0_print(
            f"[Info] MIMIC-IV-CDM MEDS training does not use validation set. "
            f"Overriding eval_strategy={training_args.eval_strategy} -> no."
        )
    training_args.eval_strategy = "no"
    training_args.load_best_model_at_end = False

    candidates, label_to_id, id_to_label = _build_label_metadata(data_args.task_name)
    tokenizer_source = model_args.tokenizer_config_path
    tokenizer = _load_clmbr_tokenizer(tokenizer_source)

    rank0_print("=" * 80)
    rank0_print("MIMIC-IV-CDM MEDS Llama Encoder Classifier Train")
    rank0_print("=" * 80)
    rank0_print(f"Model path: {model_args.model_name_or_path}")
    rank0_print(f"Tokenizer source: {tokenizer_source}")
    rank0_print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    rank0_print(f"Task: {data_args.task_name}")
    rank0_print(f"Root dir: {data_args.root_dir}")
    rank0_print(f"Max seq length: {data_args.max_seq_length}")
    rank0_print(f"Concept map dir: {data_args.concept_map_dir}")
    rank0_print(f"Freeze encoder: {model_args.freeze_encoder}")
    rank0_print(f"Use PEFT: {model_args.use_peft}")
    if model_args.use_peft:
        rank0_print(
            f"LoRA config: r={model_args.lora_r}, alpha={model_args.lora_alpha}, "
            f"dropout={model_args.lora_dropout}, target_modules={model_args.lora_target_modules}"
        )
    rank0_print("Validation usage: disabled (train split only)")
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
    rank0_print(f"Trainable parameter names: {', '.join(trainable_parameters)}")

    train_dataset = _load_source_dataset(
        data_args=data_args,
        split_name="train",
        shuffle=True,
        max_samples=data_args.max_train_samples,
    )
    rank0_print(f"Final train dataset size: {len(train_dataset)}")

    data_collator = MIMICIVCDMMEDSDataCollator(
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_seq_length=data_args.max_seq_length,
        task_name=data_args.task_name,
    )

    trainer = HeadOnlySequenceClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=None,
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
