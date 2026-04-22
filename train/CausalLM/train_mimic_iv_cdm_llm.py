import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from transformers import DataCollatorWithPadding, Trainer, set_seed
from trl import ModelConfig, SFTConfig, SFTTrainer, ScriptArguments, TrlParser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import (
    HeadOnlySequenceClassificationTrainer,
    LocalLLMScriptArguments,
    apply_template,
    build_dataset_from_source,
    build_lora_config,
    freeze_for_head_only_sequence_classification,
    load_model,
    prepare_training_args,
    rank0_print,
)
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info


MAIN_DIAGNOSIS_TASK = "MIMIC-IV-CDM Main Disease Diagnoses"


@dataclass
class MIMICIVCDMScriptArguments(LocalLLMScriptArguments, ScriptArguments):
    root_dir: str = field(
        default="/home/ma-user/sfs_turbo/Data/mimic-iv-cdm",
        metadata={"help": "Root directory for MIMIC-IV-CDM data."},
    )
    task_name: str = field(
        default="MIMIC-IV-CDM Main Disease Diagnoses",
        metadata={"help": "Task name: 'MIMIC-IV-CDM Main Disease Diagnoses' or 'MIMIC-IV-CDM ICD Code Diagnoses'."},
    )
    table_mode: str = field(
        default="text_only",
        metadata={"help": "Input mode: 'text_only' or 'table_only'."},
    )
    lazy_mode: bool = field(default=True, metadata={"help": "Load MIMIC-IV-CDM samples lazily."})
    max_seq_len: Optional[int] = field(default=8192, metadata={"help": "Optional alias for max_seq_length."})
    per_device_batch_size: Optional[int] = field(
        default=2,
        metadata={"help": "Optional alias for per_device_train_batch_size."},
    )


def _load_task_dataset(script_args, processor_or_tokenizer, split: str, shuffle_after_build: bool):
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
    dataset = build_dataset_from_source(
        base_dataset=source,
        processor_or_tokenizer=processor_or_tokenizer,
        max_seq_length=script_args.max_seq_length,
        system_prompt=script_args.system_prompt,
        shuffle=False,
    )
    if shuffle_after_build:
        dataset = dataset.shuffle(seed=42)
    return dataset


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    candidates = list(task_info["candidate"])
    label_to_id = {label: idx for idx, label in enumerate(candidates)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return candidates, label_to_id, id_to_label


def _build_classification_dataset(script_args, processor_or_tokenizer, split: str, label_to_id):
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
    tokenizer = processor_or_tokenizer.tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else processor_or_tokenizer
    model_name_or_path = tokenizer.name_or_path
    rows = []
    for index in range(len(source)):
        sample = source[index]
        prompt = apply_template(
            model_name_or_path=model_name_or_path,
            processor_or_tokenizer=processor_or_tokenizer,
            input_text=str(sample.get("input", "")),
            instruction=sample["instruction"],
            system_prompt=script_args.system_prompt,
            output_text=None,
        )
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=script_args.max_seq_length,
            return_token_type_ids=True,
        )
        tokenized["labels"] = label_to_id[str(sample["output"])]
        tokenized["idx"] = index
        rows.append(tokenized)
    return Dataset.from_list(rows)


def main():
    parser = TrlParser((MIMICIVCDMScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    if script_args.max_seq_len is not None:
        script_args.max_seq_length = script_args.max_seq_len
    if script_args.per_device_batch_size is not None:
        training_args.per_device_train_batch_size = script_args.per_device_batch_size
    if script_args.use_sequence_classification:
        training_args.remove_unused_columns = False
        training_args.seed = getattr(training_args, "seed", 42) or 42
        set_seed(training_args.seed)
    else:
        training_args = prepare_training_args(training_args)

    rank0_print("=" * 80)
    rank0_print("MIMIC-IV-CDM LLM SFT")
    rank0_print("=" * 80)
    rank0_print(f"Local model path: {model_config.model_name_or_path}")
    rank0_print(f"Task: {script_args.task_name}")
    rank0_print(f"Table mode: {script_args.table_mode}")
    rank0_print(f"Max seq len: {script_args.max_seq_length}")
    rank0_print(f"Per-device batch size: {training_args.per_device_train_batch_size}")
    rank0_print(f"Use sequence classification: {script_args.use_sequence_classification}")
    rank0_print(f"Output dir: {training_args.output_dir}")

    if script_args.use_sequence_classification:
        if script_args.task_name != MAIN_DIAGNOSIS_TASK:
            raise NotImplementedError(
                "Sequence classification mode currently supports only 'MIMIC-IV-CDM Main Disease Diagnoses'."
            )
        candidates, label_to_id, id_to_label = _build_label_metadata(script_args.task_name)
        model, processor_or_tokenizer = load_model(
            model_config.model_name_or_path,
            use_sequence_classification=True,
            num_labels=len(candidates),
            label2id=label_to_id,
            id2label=id_to_label,
        )
        model = freeze_for_head_only_sequence_classification(model)
        tokenizer = processor_or_tokenizer.tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else processor_or_tokenizer
        train_dataset = _build_classification_dataset(
            script_args=script_args,
            processor_or_tokenizer=processor_or_tokenizer,
            split="train",
            label_to_id=label_to_id,
        )
        rank0_print(f"Final train dataset size: {len(train_dataset)}")

        eval_dataset = None
        if training_args.eval_strategy != "no":
            eval_dataset = _build_classification_dataset(
                script_args=script_args,
                processor_or_tokenizer=processor_or_tokenizer,
                split="val",
                label_to_id=label_to_id,
            )
            rank0_print(f"Final validation dataset size: {len(eval_dataset)}")

        trainer = HeadOnlySequenceClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
    else:
        model, processor_or_tokenizer = load_model(model_config.model_name_or_path)

        train_dataset = _load_task_dataset(
            script_args=script_args,
            processor_or_tokenizer=processor_or_tokenizer,
            split="train",
            shuffle_after_build=True,
        )
        rank0_print(f"Final train dataset size: {len(train_dataset)}")

        eval_dataset = None
        if training_args.eval_strategy != "no":
            eval_dataset = _load_task_dataset(
                script_args=script_args,
                processor_or_tokenizer=processor_or_tokenizer,
                split="val",
                shuffle_after_build=False,
            )
            rank0_print(f"Final validation dataset size: {len(eval_dataset)}")

        peft_config = build_lora_config(model_config)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processor_or_tokenizer,
            peft_config=peft_config,
        )

    rank0_print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    rank0_print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    processor_or_tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
