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
    augment_binary_aliases,
    build_prompts_from_dataset,
    compute_sequence_classification_metrics,
    detect_model_family,
    load_model,
    load_tokenizer,
    normalize_label,
    tokenize_classification_prompts,
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
    checkpoint_dir: str = field(metadata={"help": "Path to the fine-tuned checkpoint directory."})


@dataclass
class DataArguments:
    data_dir: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular",
        metadata={"help": "Root directory for MIMIC-IV tabular data used by EHR-Bench."},
    )
    sample_info_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test CSV. Defaults to <data_dir>/task_index/test/<task_name>.csv."},
    )
    task_name: str = field(
        default="ED_Hospitalization",
        metadata={"help": f"EHR-Bench risk task. One of: {ALL_RISK_PREDICTION_TASKS}"},
    )
    table_mode: str = field(
        default="table_only",
        metadata={"help": "Input mode: 'text_only', 'table_only', or 'table_plus_rest_text'."},
    )
    lazy_mode: bool = field(default=True, metadata={"help": "Load EHR-Bench samples lazily."})
    max_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of test samples."})
    max_seq_len: int = field(default=8192, metadata={"help": "Maximum context length."})
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    output_dir: Optional[str] = field(default=None, metadata={"help": "Directory for evaluation outputs."})


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    task_type = task_info["task_type"]
    if task_type == "multi_label_classification":
        raise NotImplementedError(
            f"EncoderLM testing does not support multi-label MIMIC task '{task_name}'."
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
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    if data_args.task_name not in ALL_RISK_PREDICTION_TASKS:
        raise ValueError(
            f"Unsupported task_name '{data_args.task_name}'. "
            f"Supported tasks: {ALL_RISK_PREDICTION_TASKS}"
        )
    set_seed(data_args.seed)

    if not os.path.isdir(model_args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {model_args.checkpoint_dir}")

    sample_info_path = data_args.sample_info_path or os.path.join(
        data_args.data_dir,
        "task_index",
        "test",
        f"{data_args.task_name}.csv",
    )
    candidates, label_to_id, id_to_label = _build_label_metadata(data_args.task_name)

    print("=" * 80)
    print("EHR-Bench EncoderLM Test")
    print("=" * 80)
    print(f"Checkpoint directory: {model_args.checkpoint_dir}")
    print(f"Task: {data_args.task_name}")
    print(f"Table mode: {data_args.table_mode}")
    print(f"Max seq len: {data_args.max_seq_len}")

    model = load_model(
        model_args.checkpoint_dir,
        num_labels=len(candidates),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    tokenizer = load_tokenizer(model_args.checkpoint_dir)
    detect_model_family(model_args.checkpoint_dir)

    dataset = MIMICIV(
        root_dir=data_args.data_dir,
        sample_info_path=sample_info_path,
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode=data_args.table_mode,
        max_samples=data_args.max_samples,
    )
    print(f"test source [{data_args.task_name}, {data_args.table_mode}] size: {len(dataset)}")
    if len(dataset) == 0:
        print(f"[SKIP] Empty test dataset for task={data_args.task_name}.")
        return

    prompts, meta_list = build_prompts_from_dataset(
        dataset,
        tokenizer,
        system_prompt="",
        max_seq_length=data_args.max_seq_len,
    )
    eval_dataset = tokenize_classification_prompts(
        prompts,
        meta_list,
        tokenizer,
        label_to_id,
        data_args.max_seq_len,
        label_normalizer=normalize_label,
    )
    if len(eval_dataset) == 0:
        print(f"[SKIP] Empty tokenized dataset for task={data_args.task_name}.")
        return

    output_dir = data_args.output_dir or os.path.join(model_args.checkpoint_dir, "eval_logs")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=data_args.batch_size,
            remove_unused_columns=False,
            report_to="none",
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Starting evaluation...")
    predict_outputs = trainer.predict(eval_dataset)
    if not trainer.is_world_process_zero():
        return
    metrics_df, raw_df = compute_sequence_classification_metrics(
        logits=predict_outputs.predictions,
        labels=predict_outputs.label_ids,
        task_name=data_args.task_name,
        idx_list=[meta["idx"] for meta in meta_list],
        id2label=id_to_label,
    )

    print(metrics_df.to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    raw_path = os.path.join(output_dir, "raw_predictions.csv")
    metrics_df.to_csv(metrics_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    print(f"Raw predictions saved to {raw_path}")


if __name__ == "__main__":
    main()
