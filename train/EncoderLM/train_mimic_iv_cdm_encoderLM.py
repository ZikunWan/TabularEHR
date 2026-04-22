import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import (
    LocalEncoderLMScriptArguments,
    detect_model_family,
    freeze_for_head_only_sequence_classification,
    load_model,
    rank0_print,
)
from classification_utils import tokenize_classification_dataset
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info


MAIN_DIAGNOSIS_TASK = "MIMIC-IV-CDM Main Disease Diagnoses"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the encoder model."})


@dataclass
class EncoderMIMICIVCDMScriptArguments(LocalEncoderLMScriptArguments):
    root_dir: str = field(
        default="/home/ma-user/sfs_turbo/Data/mimic-iv-cdm",
        metadata={"help": "Root directory for MIMIC-IV-CDM data."},
    )
    task_name: str = field(
        default=MAIN_DIAGNOSIS_TASK,
        metadata={"help": "Currently only supports 'MIMIC-IV-CDM Main Disease Diagnoses'."},
    )
    table_mode: str = field(
        default="table_only",
        metadata={"help": "Input mode: 'text_only', 'table_only', or 'table_plus_rest_text'."},
    )
    lazy_mode: bool = field(default=True, metadata={"help": "Load MIMIC-IV-CDM samples lazily."})
    max_seq_len: Optional[int] = field(default=8192, metadata={"help": "Optional alias for max_seq_length."})
    per_device_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Optional alias for per_device_train_batch_size."},
    )


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    candidates = list(task_info["candidate"])
    label_to_id = {label: idx for idx, label in enumerate(candidates)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return candidates, label_to_id, id_to_label


def _load_source_dataset(script_args, split: str):
    source = MIMICIVCDM(
        root_dir=script_args.root_dir,
        split=split,
        lazy_mode=script_args.lazy_mode,
        shuffle=False,
        table_mode=script_args.table_mode,
        task_name=script_args.task_name,
        max_samples=None,
    )
    rank0_print(f"{split} source [{script_args.task_name}, {script_args.table_mode}] size: {len(source)}")
    return source


def main():
    parser = HfArgumentParser((ModelArguments, EncoderMIMICIVCDMScriptArguments, TrainingArguments))
    model_args, script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.task_name != MAIN_DIAGNOSIS_TASK:
        raise NotImplementedError(
            "EncoderLM training currently supports only 'MIMIC-IV-CDM Main Disease Diagnoses'."
        )
    if script_args.max_seq_len is not None:
        script_args.max_seq_length = script_args.max_seq_len
    if script_args.per_device_batch_size is not None:
        training_args.per_device_train_batch_size = script_args.per_device_batch_size

    training_args.remove_unused_columns = False
    training_args.save_safetensors = True
    training_args.seed = getattr(training_args, "seed", 42) or 42
    set_seed(training_args.seed)

    detect_model_family(model_args.model_name_or_path)
    candidates, label_to_id, id_to_label = _build_label_metadata(script_args.task_name)

    rank0_print("=" * 80)
    rank0_print("MIMIC-IV-CDM EncoderLM Train")
    rank0_print("=" * 80)
    rank0_print(f"Model path: {model_args.model_name_or_path}")
    rank0_print(f"Task: {script_args.task_name}")
    rank0_print(f"Table mode: {script_args.table_mode}")
    rank0_print(f"Max seq len: {script_args.max_seq_length}")
    rank0_print(f"Per-device batch size: {training_args.per_device_train_batch_size}")
    rank0_print(f"Output dir: {training_args.output_dir}")

    model, tokenizer = load_model(
        model_args.model_name_or_path,
        num_labels=len(candidates),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model = freeze_for_head_only_sequence_classification(model)

    train_source = _load_source_dataset(script_args, split="train")
    train_dataset = tokenize_classification_dataset(
        source_dataset=train_source,
        tokenizer=tokenizer,
        model_name_or_path=model_args.model_name_or_path,
        max_seq_length=script_args.max_seq_length,
        system_prompt=script_args.system_prompt,
        label_to_id=label_to_id,
        task_name=script_args.task_name,
        dataset_name="MIMIC-IV-CDM",
    )
    rank0_print(f"Final train dataset size: {len(train_dataset)}")

    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_source = _load_source_dataset(script_args, split="val")
        eval_dataset = tokenize_classification_dataset(
            source_dataset=eval_source,
            tokenizer=tokenizer,
            model_name_or_path=model_args.model_name_or_path,
            max_seq_length=script_args.max_seq_length,
            system_prompt=script_args.system_prompt,
            label_to_id=label_to_id,
            task_name=script_args.task_name,
            dataset_name="MIMIC-IV-CDM",
        )
        rank0_print(f"Final validation dataset size: {len(eval_dataset)}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    rank0_print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    rank0_print(f"Saving model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
