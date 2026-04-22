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

from classification_utils import augment_binary_aliases, normalize_label, tokenize_classification_dataset
from common import (
    LocalEncoderLMScriptArguments,
    detect_model_family,
    freeze_for_head_only_sequence_classification,
    load_model,
    rank0_print,
)
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info


ALL_RISK_PREDICTION_TASKS = [
    "ED_Hospitalization",
    "ED_Inpatient_Mortality",
    "ED_ICU_Tranfer_12hour",
    "ED_Reattendance_3day",
    "ED_Critical_Outcomes",
    "Readmission_30day",
    "Readmission_60day",
    "Inpatient_Mortality",
    "LengthOfStay_3day",
    "LengthOfStay_7day",
    "ICU_Mortality_1day",
    "ICU_Mortality_2day",
    "ICU_Mortality_3day",
    "ICU_Mortality_7day",
    "ICU_Mortality_14day",
    "ICU_Stay_7day",
    "ICU_Stay_14day",
    "ICU_Readmission",
]


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the encoder model."})


@dataclass
class EHRBenchEncoderScriptArguments(LocalEncoderLMScriptArguments):
    data_dir: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular",
        metadata={"help": "Root directory for MIMIC-IV tabular data used by EHR-Bench."},
    )
    task_name: str = field(
        default="ED_Hospitalization",
        metadata={"help": f"EHR-Bench risk task. One of: {ALL_RISK_PREDICTION_TASKS}"},
    )
    train_sample_info_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train CSV. Defaults to <data_dir>/task_index/train/<task_name>.csv."},
    )
    val_sample_info_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to val CSV. Defaults to <data_dir>/task_index/val/<task_name>.csv."},
    )
    table_mode: str = field(
        default="table_only",
        metadata={"help": "Input mode: 'text_only', 'table_only', or 'table_plus_rest_text'."},
    )
    lazy_mode: bool = field(default=True, metadata={"help": "Load EHR-Bench samples lazily."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of train samples."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of validation samples."})
    max_seq_len: Optional[int] = field(default=8192, metadata={"help": "Optional alias for max_seq_length."})
    per_device_batch_size: Optional[int] = field(
        default=8,
        metadata={"help": "Optional alias for per_device_train_batch_size."},
    )


def _load_source_dataset(script_args, sample_info_path: str, split_name: str, shuffle: bool, max_samples: Optional[int]):
    dataset = MIMICIV(
        root_dir=script_args.data_dir,
        sample_info_path=sample_info_path,
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
            f"EncoderLM training does not support multi-label MIMIC task '{task_name}'."
        )

    if "candidate" in task_info:
        candidates = [str(candidate) for candidate in task_info["candidate"]]
    elif task_type == "binary_classification":
        candidates = ["0", "1"]
    elif task_type == "multi_class_classification" and task_info.get("num_classes") is not None:
        candidates = [str(index) for index in range(int(task_info["num_classes"]))]
    else:
        raise ValueError(f"Unsupported MIMIC task_type '{task_type}' for task '{task_name}'.")

    candidates = [normalize_label(candidate) for candidate in candidates]
    label_to_id = {label: idx for idx, label in enumerate(candidates)}
    id_to_label = {idx: label for idx, label in enumerate(candidates)}
    if task_type == "binary_classification":
        augment_binary_aliases(label_to_id)
    return candidates, label_to_id, id_to_label


def main():
    parser = HfArgumentParser((ModelArguments, EHRBenchEncoderScriptArguments, TrainingArguments))
    model_args, script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.task_name not in ALL_RISK_PREDICTION_TASKS:
        raise ValueError(
            f"Unsupported task_name '{script_args.task_name}'. "
            f"Supported tasks: {ALL_RISK_PREDICTION_TASKS}"
        )
    if script_args.max_seq_len is not None:
        script_args.max_seq_length = script_args.max_seq_len
    if script_args.per_device_batch_size is not None:
        training_args.per_device_train_batch_size = script_args.per_device_batch_size

    train_info_path = script_args.train_sample_info_path or os.path.join(
        script_args.data_dir,
        "task_index",
        "train",
        f"{script_args.task_name}.csv",
    )
    val_info_path = script_args.val_sample_info_path or os.path.join(
        script_args.data_dir,
        "task_index",
        "val",
        f"{script_args.task_name}.csv",
    )

    training_args.remove_unused_columns = False
    training_args.save_safetensors = True
    training_args.seed = getattr(training_args, "seed", 42) or 42
    set_seed(training_args.seed)

    detect_model_family(model_args.model_name_or_path)
    candidates, label_to_id, id_to_label = _build_label_metadata(script_args.task_name)

    rank0_print("=" * 80)
    rank0_print("EHR-Bench EncoderLM Train")
    rank0_print("=" * 80)
    rank0_print(f"Model path: {model_args.model_name_or_path}")
    rank0_print(f"Task: {script_args.task_name}")
    rank0_print(f"Train CSV: {train_info_path}")
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

    train_source = _load_source_dataset(
        script_args=script_args,
        sample_info_path=train_info_path,
        split_name="train",
        shuffle=True,
        max_samples=script_args.max_train_samples,
    )
    train_dataset = tokenize_classification_dataset(
        source_dataset=train_source,
        tokenizer=tokenizer,
        model_name_or_path=model_args.model_name_or_path,
        max_seq_length=script_args.max_seq_length,
        system_prompt=script_args.system_prompt,
        label_to_id=label_to_id,
        task_name=script_args.task_name,
        dataset_name="EHR-Bench",
        label_normalizer=normalize_label,
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
        eval_dataset = tokenize_classification_dataset(
            source_dataset=eval_source,
            tokenizer=tokenizer,
            model_name_or_path=model_args.model_name_or_path,
            max_seq_length=script_args.max_seq_length,
            system_prompt=script_args.system_prompt,
            label_to_id=label_to_id,
            task_name=script_args.task_name,
            dataset_name="EHR-Bench",
            label_normalizer=normalize_label,
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
