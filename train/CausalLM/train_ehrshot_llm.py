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
from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info


@dataclass
class EHRShotScriptArguments(LocalLLMScriptArguments, ScriptArguments):
    root_dir: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT",
        metadata={"help": "Root directory for EHRShot data."},
    )
    train_info_path: Optional[str] = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT/index/ehrshot_train.csv",
        metadata={"help": "Path to EHRShot training index CSV. Defaults to <root_dir>/index/ehrshot_train.csv."},
    )
    val_info_path: Optional[str] = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT/index/ehrshot_val.csv",
        metadata={"help": "Path to EHRShot validation index CSV. Defaults to <root_dir>/index/ehrshot_val.csv."},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional single task name. If omitted, train on all tasks in the index file."},
    )
    table_mode: str = field(
        default="text_only",
        metadata={"help": "Input mode: 'text_only', 'table_only', or 'table_plus_rest_text'."},
    )
    lazy_mode: bool = field(default=True, metadata={"help": "Load EHRShot samples lazily."})
    max_train_samples: Optional[int] = field(default=50000, metadata={"help": "Maximum number of train samples."})
    max_eval_samples: Optional[int] = field(default=5000, metadata={"help": "Maximum number of validation samples."})


def _load_source_dataset(script_args, sample_info_path: str, split_name: str, max_samples: Optional[int]):
    dataset = EHRSHOTDataset(
        root_dir=script_args.root_dir,
        sample_info_path=sample_info_path,
        task_name=script_args.task_name,
        lazy_mode=script_args.lazy_mode,
        table_mode=script_args.table_mode,
        max_samples=max_samples,
    )
    rank0_print(f"{split_name} source [{script_args.task_name or 'all'}, {script_args.table_mode}] size: {len(dataset)}")
    return dataset


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
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return candidates, label_to_id, id_to_label


def _build_classification_dataset(
    source_dataset,
    processor_or_tokenizer,
    model_name_or_path: str,
    max_seq_length: int,
    system_prompt: str,
    label_to_id,
    task_name: str,
):
    tokenizer = processor_or_tokenizer.tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else processor_or_tokenizer
    rows = []
    for index in range(len(source_dataset)):
        sample = source_dataset[index]
        prompt = apply_template(
            model_name_or_path=model_name_or_path,
            processor_or_tokenizer=processor_or_tokenizer,
            input_text=str(sample.get("input", "")),
            instruction=sample["instruction"],
            system_prompt=system_prompt,
            output_text=None,
        )
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            return_token_type_ids=True,
        )
        label = str(sample["output"])
        if label not in label_to_id:
            raise ValueError(f"Unexpected label '{label}' for EHRShot task '{task_name}'.")
        tokenized["labels"] = label_to_id[label]
        tokenized["idx"] = index
        rows.append(tokenized)
    return Dataset.from_list(rows)


def main():
    parser = TrlParser((EHRShotScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    if script_args.use_sequence_classification:
        training_args.remove_unused_columns = False
        training_args.seed = getattr(training_args, "seed", 42) or 42
        set_seed(training_args.seed)
    else:
        training_args = prepare_training_args(training_args)

    train_info_path = script_args.train_info_path or os.path.join(script_args.root_dir, "index", "ehrshot_train.csv")
    val_info_path = script_args.val_info_path or os.path.join(script_args.root_dir, "index", "ehrshot_val.csv")

    rank0_print("=" * 80)
    rank0_print("EHRShot LLM SFT")
    rank0_print("=" * 80)
    rank0_print(f"Local model path: {model_config.model_name_or_path}")
    rank0_print(f"Train index: {train_info_path}")
    rank0_print(f"Task filter: {script_args.task_name or 'all'}")
    rank0_print(f"Use sequence classification: {script_args.use_sequence_classification}")

    if script_args.use_sequence_classification:
        if not script_args.task_name:
            raise ValueError("Sequence classification mode requires --task_name for EHRShot.")

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

        train_source = _load_source_dataset(
            script_args=script_args,
            sample_info_path=train_info_path,
            split_name="train",
            max_samples=script_args.max_train_samples,
        )
        train_dataset = _build_classification_dataset(
            source_dataset=train_source,
            processor_or_tokenizer=processor_or_tokenizer,
            model_name_or_path=model_config.model_name_or_path,
            max_seq_length=script_args.max_seq_length,
            system_prompt=script_args.system_prompt,
            label_to_id=label_to_id,
            task_name=script_args.task_name,
        )

        eval_dataset = None
        if training_args.eval_strategy != "no":
            eval_source = _load_source_dataset(
                script_args=script_args,
                sample_info_path=val_info_path,
                split_name="validation",
                max_samples=script_args.max_eval_samples,
            )
            eval_dataset = _build_classification_dataset(
                source_dataset=eval_source,
                processor_or_tokenizer=processor_or_tokenizer,
                model_name_or_path=model_config.model_name_or_path,
                max_seq_length=script_args.max_seq_length,
                system_prompt=script_args.system_prompt,
                label_to_id=label_to_id,
                task_name=script_args.task_name,
            )

        trainer = HeadOnlySequenceClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        )
    else:
        model, processor_or_tokenizer = load_model(model_config.model_name_or_path)

        train_source = _load_source_dataset(
            script_args=script_args,
            sample_info_path=train_info_path,
            split_name="train",
            max_samples=script_args.max_train_samples,
        )
        train_dataset = build_dataset_from_source(
            base_dataset=train_source,
            processor_or_tokenizer=processor_or_tokenizer,
            max_seq_length=script_args.max_seq_length,
            system_prompt=script_args.system_prompt,
            shuffle=True,
        )

        eval_dataset = None
        if training_args.eval_strategy != "no":
            eval_source = _load_source_dataset(
                script_args=script_args,
                sample_info_path=val_info_path,
                split_name="validation",
                max_samples=script_args.max_eval_samples,
            )
            eval_dataset = build_dataset_from_source(
                base_dataset=eval_source,
                processor_or_tokenizer=processor_or_tokenizer,
                max_seq_length=script_args.max_seq_length,
                system_prompt=script_args.system_prompt,
                shuffle=False,
            )

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
