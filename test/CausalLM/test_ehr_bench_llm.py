import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams
from vllm.config import PoolerConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import (
    build_prompts_from_dataset,
    cleanup_materialized_vllm_model,
    compute_sequence_classification_metrics,
    load_tokenizer,
    materialize_vllm_sequence_classification_model,
    resolve_tensor_parallel_size,
    resolve_vllm_model_and_lora,
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
class EHRBenchTestArguments:
    model_path: str = field(metadata={"help": "Path to trained model or adapter."})
    base_model_path: Optional[str] = field(default=None, metadata={"help": "Optional base model path when model_path is a LoRA adapter."})
    data_dir: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular",
        metadata={"help": "Root directory for MIMIC-IV tabular data used by EHR-Bench."},
    )
    sample_info_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to test CSV. Defaults to <data_dir>/task_index/test/<task_name>.csv."},
    )
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional task filter. Required in sequence-classification mode."},
    )
    max_samples: Optional[int] = field(default=None, metadata={"help": "Maximum number of test samples."})
    batch_size: int = field(default=16, metadata={"help": "Eval batch size for sequence classification."})
    table_mode: str = field(default="text_only", metadata={"help": "Input mode: text_only/table_only/table_plus_rest_text."})
    max_seq_len: int = field(default=8192, metadata={"help": "Maximum context length."})
    max_new_tokens: int = field(default=32, metadata={"help": "Max generated tokens for generative evaluation."})
    tp_size: int = field(default=1, metadata={"help": "Tensor parallel size for vLLM."})
    use_sequence_classification: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate via AutoModelForSequenceClassification."},
    )
    output_dir: Optional[str] = field(default=None, metadata={"help": "Directory for evaluation outputs."})


def _load_test_dataset(script_args: EHRBenchTestArguments):
    if script_args.task_name and script_args.task_name not in ALL_RISK_PREDICTION_TASKS:
        raise ValueError(
            f"Unsupported task_name '{script_args.task_name}'. "
            f"Supported tasks: {ALL_RISK_PREDICTION_TASKS}"
        )
    sample_info_path = script_args.sample_info_path or os.path.join(
        script_args.data_dir,
        "task_index",
        "test",
        f"{script_args.task_name}.csv",
    )
    dataset = MIMICIV(
        root_dir=script_args.data_dir,
        sample_info_path=sample_info_path,
        lazy_mode=True,
        shuffle=False,
        table_mode=script_args.table_mode,
        max_samples=script_args.max_samples,
    )
    print(f"test source [{script_args.task_name or 'all'}, {script_args.table_mode}] size: {len(dataset)}")
    return dataset


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    task_type = task_info["task_type"]
    if task_type == "multi_label_classification":
        return task_info, None, None, None
    if "candidate" in task_info:
        candidates = [str(candidate) for candidate in task_info["candidate"]]
    elif task_type == "binary_classification":
        candidates = ["0", "1"]
    elif task_type == "multi_class_classification" and task_info.get("num_classes") is not None:
        candidates = [str(index) for index in range(int(task_info["num_classes"]))]
    else:
        raise ValueError(f"Unsupported MIMIC task_type '{task_type}' for task '{task_name}'.")
    label_to_id = {label: idx for idx, label in enumerate(candidates)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return task_info, candidates, label_to_id, id_to_label


def _tokenize_classification_prompts(prompts, meta_list, tokenizer, label_to_id, max_seq_len: int):
    from datasets import Dataset

    rows = []
    for prompt, meta in zip(prompts, meta_list):
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_len,
            return_token_type_ids=True,
        )
        tokenized["labels"] = label_to_id[str(meta["label"])]
        tokenized["idx"] = meta["idx"]
        rows.append(tokenized)
    return Dataset.from_list(rows)


def _extract_answer(generated_text: str, candidates):
    text = generated_text.strip()
    if candidates is None:
        return text
    lowered = text.lower()
    for candidate in candidates:
        candidate_str = str(candidate)
        if lowered == candidate_str.lower():
            return candidate_str
    for candidate in candidates:
        candidate_str = str(candidate)
        if re.search(rf"\b{re.escape(candidate_str.lower())}\b", lowered):
            return candidate_str
    first_token = text.split()[0] if text else ""
    for candidate in candidates:
        candidate_str = str(candidate)
        if first_token.lower() == candidate_str.lower():
            return candidate_str
    return str(candidates[0]) if candidates else ""


def _compute_generated_metrics(raw_records):
    raw_df = pd.DataFrame(raw_records)
    metric_rows = []
    for task_name, group in raw_df.groupby("task"):
        task_type = group["task_type"].iloc[0]
        labels = group["label"].astype(str).tolist()
        preds = group["pred"].astype(str).tolist()
        count = int(len(group))

        if task_type == "binary_classification":
            acc = accuracy_score(labels, preds)
            macro_f1 = f1_score(labels, preds, average="macro")
            try:
                auroc = roc_auc_score([int(item) for item in labels], [int(item) for item in preds])
            except Exception:
                auroc = 0.5
            metric_rows.append(
                {
                    "Task": task_name,
                    "Count": count,
                    "Metric": "AUROC",
                    "Value": float(auroc),
                    "Accuracy": float(acc),
                    "Macro F1": float(macro_f1),
                }
            )
        elif task_type == "multi_class_classification":
            acc = accuracy_score(labels, preds)
            macro_f1 = f1_score(labels, preds, average="macro")
            metric_rows.append(
                {
                    "Task": task_name,
                    "Count": count,
                    "Metric": "Accuracy",
                    "Value": float(acc),
                    "Accuracy": float(acc),
                    "Macro F1": float(macro_f1),
                }
            )
        else:
            raise ValueError(f"Unsupported task_type '{task_type}' during evaluation.")
    return pd.DataFrame(metric_rows), raw_df


def main():
    parser = HfArgumentParser(EHRBenchTestArguments)
    (script_args,) = parser.parse_args_into_dataclasses()

    print("=" * 80)
    print("EHR-Bench LLM Test")
    print("=" * 80)
    print(f"Model path: {script_args.model_path}")
    print(f"Task: {script_args.task_name or 'all'}")
    print(f"Table mode: {script_args.table_mode}")
    print(f"Max seq len: {script_args.max_seq_len}")
    print(f"Use sequence classification: {script_args.use_sequence_classification}")

    dataset = _load_test_dataset(script_args)

    if script_args.use_sequence_classification:
        if not script_args.task_name:
            raise ValueError("Sequence classification mode requires --task_name for EHR-Bench evaluation.")
        task_info, candidates, label_to_id, id_to_label = _build_label_metadata(script_args.task_name)
        if task_info["task_type"] == "multi_label_classification":
            raise NotImplementedError(
                f"Sequence classification mode does not support multi-label task '{script_args.task_name}'."
            )
        save_dir = script_args.output_dir or os.path.join(script_args.model_path, "eval_logs")
        vllm_model_path = None
        was_materialized = False
        llm = None
        try:
            vllm_model_path, tokenizer_path, was_materialized = materialize_vllm_sequence_classification_model(
                script_args.model_path,
                num_labels=len(candidates),
                label2id=label_to_id,
                id2label=id_to_label,
            )
            tokenizer_or_processor = load_tokenizer(
                tokenizer_path,
                use_sequence_classification=True,
            )
            prompts, meta_list = build_prompts_from_dataset(
                dataset,
                tokenizer_or_processor,
                system_prompt="",
                max_seq_length=script_args.max_seq_len,
            )
            llm = LLM(
                model=vllm_model_path,
                tokenizer=tokenizer_path,
                runner="pooling",
                convert="classify",
                pooler_config=PoolerConfig(),
                tensor_parallel_size=resolve_tensor_parallel_size(script_args.tp_size),
                trust_remote_code=True,
                max_model_len=script_args.max_seq_len,
                gpu_memory_utilization=0.85,
            )
            outputs = llm.classify(prompts, use_tqdm=True)
        finally:
            if llm is not None:
                del llm
            if vllm_model_path is not None:
                cleanup_materialized_vllm_model(vllm_model_path, was_materialized)

        probs = np.array([output.outputs.probs for output in outputs], dtype=np.float64)
        labels = np.array([label_to_id[str(meta["label"])] for meta in meta_list], dtype=np.int64)
        logits = np.log(np.clip(probs, 1e-12, 1.0))
        metrics_df, raw_df = compute_sequence_classification_metrics(
            logits=logits,
            labels=labels,
            task_name=script_args.task_name,
            idx_list=[meta["idx"] for meta in meta_list],
            id2label=id_to_label,
        )
    else:
        vllm_model_path, tokenizer_path, lora_request = resolve_vllm_model_and_lora(
            script_args.model_path,
            script_args.base_model_path,
        )
        tokenizer_or_processor = load_tokenizer(tokenizer_path)
        prompts, _ = build_prompts_from_dataset(
            dataset,
            tokenizer_or_processor,
            system_prompt="",
            max_seq_length=script_args.max_seq_len,
        )
        llm = LLM(
            model=vllm_model_path,
            tensor_parallel_size=script_args.tp_size,
            trust_remote_code=True,
            max_model_len=script_args.max_seq_len,
            gpu_memory_utilization=0.85,
            enable_lora=lora_request is not None,
        )
        outputs = llm.generate(
            prompts,
            SamplingParams(
                temperature=0,
                max_tokens=script_args.max_new_tokens,
                logprobs=None,
            ),
            lora_request=lora_request,
        )
        raw_records = []
        for index, output in enumerate(outputs):
            sample = dataset[index]
            task_name = script_args.task_name or str(
                dataset.sample_info[index].get("task", dataset.sample_info[index].get("task_name", "unknown"))
            )
            task_info, candidates, _, _ = _build_label_metadata(task_name)
            generated_text = output.outputs[0].text.strip() if output.outputs else ""
            pred = _extract_answer(generated_text, candidates)
            raw_records.append(
                {
                    "idx": index,
                    "task": task_name,
                    "task_type": task_info["task_type"],
                    "label": str(sample["output"]),
                    "pred": pred,
                    "generated_text": generated_text,
                }
            )
        metrics_df, raw_df = _compute_generated_metrics(raw_records)
        save_dir = script_args.output_dir or os.path.join(script_args.model_path, "eval_logs")

    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.csv")
    raw_path = os.path.join(save_dir, "raw_predictions.csv")
    metrics_df.to_csv(metrics_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    print(f"Raw predictions saved to {raw_path}")
    print(f"Results saved to {metrics_path}")


if __name__ == "__main__":
    main()
