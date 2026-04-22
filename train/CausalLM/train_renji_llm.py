import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset
from trl import ModelConfig, SFTConfig, SFTTrainer, ScriptArguments, TrlParser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import (
    LocalLLMScriptArguments,
    build_dataset_from_source,
    build_lora_config,
    load_model,
    prepare_training_args,
    rank0_print,
)
from dataset.renji.renji_dataset import RenjiDataset


@dataclass
class RenjiScriptArguments(LocalLLMScriptArguments, ScriptArguments):
    root_dir: str = field(default="./data/Renji", metadata={"help": "Root directory for Renji data."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of train samples."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of validation samples."})
    target_metrics: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated metrics, e.g. ALT,AST,TB."},
    )
    target_prediction_points: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated prediction points, e.g. day14,day30."},
    )
    target_windows: Optional[str] = field(
        default=None,
        metadata={"help": "Backward-compatible alias for target_prediction_points."},
    )
    dataset_type: str = field(
        default="map",
        metadata={"help": "Use 'map' for cached HF Dataset or 'iterable' for streaming IterableDataset."},
    )
    cache_dir: Optional[str] = field(
        default="./data/cache/renji_sft",
        metadata={"help": "Directory used to cache processed Renji map datasets. Set to None to disable cache."},
    )
    force_regenerate: bool = field(
        default=False,
        metadata={"help": "Ignore any cached dataset and rebuild it."},
    )


def _parse_csv_arg(value: Optional[str]):
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def _get_cache_key(script_args, model_name: str):
    config = {
        "root_dir": script_args.root_dir,
        "max_train_samples": script_args.max_train_samples,
        "max_seq_length": script_args.max_seq_length,
        "target_metrics": script_args.target_metrics,
        "target_prediction_points": script_args.target_prediction_points or script_args.target_windows,
        "model_name": model_name,
    }
    return hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _get_cache_path(cache_dir: Optional[str], split: str, cache_key: str):
    if cache_dir is None:
        return None
    return os.path.join(cache_dir, f"{split}_{cache_key}")


def main():
    parser = TrlParser((RenjiScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    training_args = prepare_training_args(training_args)

    target_metrics = _parse_csv_arg(script_args.target_metrics)
    target_prediction_points = _parse_csv_arg(script_args.target_prediction_points or script_args.target_windows)
    if target_metrics:
        rank0_print(f"Filtering metrics: {target_metrics}")
    if target_prediction_points:
        rank0_print(f"Filtering prediction points: {target_prediction_points}")

    rank0_print("=" * 80)
    rank0_print("Renji LLM SFT")
    rank0_print("=" * 80)
    rank0_print(f"Local model path: {model_config.model_name_or_path}")
    rank0_print(f"Dataset type: {script_args.dataset_type}")

    model, processor_or_tokenizer = load_model(model_config.model_name_or_path)

    train_source = RenjiDataset(
        root_dir=script_args.root_dir,
        split="train",
        max_samples=script_args.max_train_samples,
        table_mode="text_only",
        target_metrics=target_metrics,
        target_prediction_points=target_prediction_points,
        shuffle=False,
        task_mode="single",
    )
    train_size = len(train_source)
    rank0_print(f"Train source size: {train_size}")

    cache_key = _get_cache_key(script_args, model_config.model_name_or_path)
    train_cache_path = _get_cache_path(script_args.cache_dir, "train", cache_key)
    use_cache = (
        script_args.dataset_type == "map"
        and train_cache_path is not None
        and os.path.exists(train_cache_path)
        and not script_args.force_regenerate
    )

    if use_cache:
        rank0_print(f"Loading cached training dataset from: {train_cache_path}")
        train_dataset = Dataset.load_from_disk(train_cache_path)
    else:
        train_dataset = build_dataset_from_source(
            base_dataset=train_source,
            processor_or_tokenizer=processor_or_tokenizer,
            max_seq_length=script_args.max_seq_length,
            system_prompt=script_args.system_prompt,
            shuffle=True,
            iterable=script_args.dataset_type == "iterable",
        )
        if script_args.dataset_type == "map" and train_cache_path is not None:
            os.makedirs(os.path.dirname(train_cache_path), exist_ok=True)
            rank0_print(f"Saving processed training dataset to cache: {train_cache_path}")
            train_dataset.save_to_disk(train_cache_path)
            metadata_path = train_cache_path + "_metadata.json"
            metadata = {
                "root_dir": script_args.root_dir,
                "max_train_samples": script_args.max_train_samples,
                "max_seq_length": script_args.max_seq_length,
                "target_metrics": target_metrics,
                "target_prediction_points": target_prediction_points,
                "model_name": model_config.model_name_or_path,
                "dataset_size": len(train_dataset),
                "cache_key": cache_key,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    if script_args.dataset_type == "iterable":
        world_size = max(1, torch.cuda.device_count())
        global_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * world_size
        )
        steps_per_epoch = max(1, math.ceil(train_size / global_batch_size))
        training_args.max_steps = int(training_args.num_train_epochs * steps_per_epoch)
        rank0_print(f"Iterable dataset detected, setting max_steps={training_args.max_steps}")

    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_source = RenjiDataset(
            root_dir=script_args.root_dir,
            split="val",
            max_samples=script_args.max_eval_samples,
            table_mode="text_only",
            target_metrics=target_metrics,
            target_prediction_points=target_prediction_points,
            shuffle=False,
            task_mode="single",
        )
        rank0_print(f"Validation source size: {len(eval_source)}")
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
