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
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info


@dataclass
class EICUScriptArguments(LocalLLMScriptArguments, ScriptArguments):
    root_dir: str = field(
        default="/home/ma-user/sfs_turbo/Data/eicu-crd/2.0",
        metadata={"help": "Root directory for raw eICU data."},
    )
    processed_dir: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed",
        metadata={"help": "Processed eICU directory containing sample_info_*.json and patients/."},
    )
    train_info_path: Optional[str] = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed/sample_info_train.json",
        metadata={"help": "Path to train sample_info JSON. Defaults to <processed_dir>/sample_info_train.json."},
    )
    val_info_path: Optional[str] = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed/sample_info_val.json",
        metadata={"help": "Path to val sample_info JSON. Defaults to <processed_dir>/sample_info_val.json."},
    )
    task_name: str = field(
        default="mortality",
        metadata={"help": "eICU task name, e.g. mortality/readmission/los_3day/diagnosis."},
    )
    table_mode: str = field(
        default="text_only",
        metadata={"help": "Input mode: 'text_only', 'table_only', or 'table_plus_rest_text'."},
    )
    lazy_mode: bool = field(default=True, metadata={"help": "Load eICU samples lazily."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of train samples."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of validation samples."})


def _load_source_dataset(script_args, sample_info_path: str, split_name: str, shuffle: bool, max_samples: Optional[int]):
    dataset = EICUDataset(
        root_dir=script_args.root_dir,
        processed_dir=script_args.processed_dir,
        sample_info_path=sample_info_path,
        task_name=script_args.task_name,
        lazy_mode=script_args.lazy_mode,
        shuffle=shuffle,
        table_mode=script_args.table_mode,
        max_samples=max_samples,
    )
    rank0_print(f"{split_name} source [{script_args.task_name}, {script_args.table_mode}] size: {len(dataset)}")
    return dataset


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    task_type = task_info["task_type"]
    if task_type == "multi_label_classification":
        raise NotImplementedError(
            f"Sequence classification mode does not currently support multi-label eICU task '{task_name}'."
        )

    # eICU preprocessing stores multiclass targets as category codes (for example
    # final_acuity/imminent_discharge are written as 0..N-1 in labeled_cohorts).
    # Sequence classification therefore needs numeric label ids even if we also
    # keep a human-readable candidate list in task metadata.
    if task_type == "multi_class_classification":
        candidates = [str(index) for index in range(int(task_info["num_classes"]))]
    elif "candidate" in task_info:
        candidates = [str(candidate) for candidate in task_info["candidate"]]
    elif task_type == "binary_classification":
        candidates = ["0", "1"]
    else:
        raise ValueError(f"Unsupported eICU task_type '{task_type}' for task '{task_name}'.")

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
):
    tokenizer = processor_or_tokenizer.tokenizer if hasattr(processor_or_tokenizer, "tokenizer") else processor_or_tokenizer
    from tqdm import trange
    rows = []
    for index in trange(len(source_dataset), desc="Building Classification Dataset"):
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
            raise ValueError(f"Unexpected label '{label}' for eICU task '{source_dataset.task_name}'.")
        tokenized["labels"] = label_to_id[label]
        tokenized["idx"] = index
        rows.append(tokenized)
    return Dataset.from_list(rows)


def main():
    parser = TrlParser((EICUScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    if script_args.use_sequence_classification:
        training_args.remove_unused_columns = False
        training_args.seed = getattr(training_args, "seed", 42) or 42
        set_seed(training_args.seed)
    else:
        training_args = prepare_training_args(training_args)

    train_info_path = script_args.train_info_path or os.path.join(script_args.processed_dir, "sample_info_train.json")
    val_info_path = script_args.val_info_path or os.path.join(script_args.processed_dir, "sample_info_val.json")

    rank0_print("=" * 80)
    rank0_print("eICU LLM SFT")
    rank0_print("=" * 80)
    rank0_print(f"Local model path: {model_config.model_name_or_path}")
    rank0_print(f"Task: {script_args.task_name}")
    rank0_print(f"Train info: {train_info_path}")
    rank0_print(f"Use sequence classification: {script_args.use_sequence_classification}")

    if script_args.use_sequence_classification:
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
            shuffle=True,
            max_samples=script_args.max_train_samples,
        )
        train_dataset = _build_classification_dataset(
            source_dataset=train_source,
            processor_or_tokenizer=processor_or_tokenizer,
            model_name_or_path=model_config.model_name_or_path,
            max_seq_length=script_args.max_seq_length,
            system_prompt=script_args.system_prompt,
            label_to_id=label_to_id,
        )
        rank0_print(f"Final train dataset size: {len(train_dataset)}")

        eval_dataset = None
        if training_args.eval_strategy != "no":
            eval_source = _load_source_dataset(
                script_args=script_args,
                sample_info_path=val_info_path,
                split_name="validation",
                shuffle=False,
                max_samples=script_args.max_eval_samples,
            )
            eval_dataset = _build_classification_dataset(
                source_dataset=eval_source,
                processor_or_tokenizer=processor_or_tokenizer,
                model_name_or_path=model_config.model_name_or_path,
                max_seq_length=script_args.max_seq_length,
                system_prompt=script_args.system_prompt,
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

        train_source = _load_source_dataset(
            script_args=script_args,
            sample_info_path=train_info_path,
            split_name="train",
            shuffle=True,
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
                shuffle=False,
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
