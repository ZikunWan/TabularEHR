import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
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
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info


@dataclass
class MIMICIVCDMTestArguments:
    model_path: str = field(metadata={"help": "Path to trained model or adapter."})
    base_model_path: Optional[str] = field(default=None, metadata={"help": "Optional base model path when model_path is a LoRA adapter."})
    root_dir: str = field(
        default="/home/ma-user/sfs_turbo/Data/mimic-iv-cdm",
        metadata={"help": "Root directory for MIMIC-IV-CDM data."},
    )
    task_name: str = field(
        default="MIMIC-IV-CDM Main Disease Diagnoses",
        metadata={"help": "Task name: 'MIMIC-IV-CDM Main Disease Diagnoses' or 'MIMIC-IV-CDM ICD Code Diagnoses'."},
    )
    table_mode: str = field(default="text_only", metadata={"help": "Input mode: 'text_only' or 'table_only'."})
    lazy_mode: bool = field(default=False, metadata={"help": "Load dataset lazily."})
    max_seq_len: int = field(default=32768, metadata={"help": "Maximum context length."})
    parallel_mode: str = field(default="tp", metadata={"help": "Parallel mode: 'dp', 'tp', or 'pp'."})
    tp_size: int = field(default=0, metadata={"help": "Tensor parallel size for vLLM. 0 means use all visible GPUs."})
    pp_size: int = field(default=1, metadata={"help": "Pipeline parallel size for vLLM."})
    score_mode: str = field(
        default="logprobs",
        metadata={"help": "Main disease scoring mode: 'logprobs' or 'prompt_logprobs'."},
    )
    use_sequence_classification: bool = field(
        default=False,
        metadata={"help": "Whether to load the model via AutoModelForSequenceClassification instead of vLLM."},
    )
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size for sequence classification."})
    output_dir: str = field(default=None, metadata={"help": "Directory for evaluation outputs."})

def _load_test_dataset(script_args: MIMICIVCDMTestArguments):
    val_dataset = MIMICIVCDM(
        root_dir=script_args.root_dir,
        split="val",
        lazy_mode=script_args.lazy_mode,
        shuffle=False,
        table_mode=script_args.table_mode,
        task_name=script_args.task_name,
        max_samples=None,
    )
    test_dataset = MIMICIVCDM(
        root_dir=script_args.root_dir,
        split="test",
        lazy_mode=script_args.lazy_mode,
        shuffle=False,
        table_mode=script_args.table_mode,
        task_name=script_args.task_name,
        max_samples=None,
    )
    val_size = len(val_dataset)
    test_size = len(test_dataset)

    test_dataset.list_data = val_dataset.list_data + test_dataset.list_data
    if hasattr(test_dataset, "data") and hasattr(val_dataset, "data"):
        test_dataset.data = val_dataset.data + test_dataset.data

    print(
        f"merged eval source [val+test, {script_args.task_name}, {script_args.table_mode}] "
        f"size: {len(test_dataset)} (val={val_size}, test={test_size})"
    )
    return test_dataset


def _build_label_metadata(task_name: str):
    task_info = get_task_info()[task_name]
    candidates = list(task_info["candidate"])
    label_to_id = {label: idx for idx, label in enumerate(candidates)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return candidates, label_to_id, id_to_label


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

def _get_main_disease_candidates():
    return ["appendicitis", "cholecystitis", "diverticulitis", "pancreatitis"]


def _slice_chunk(items, chunk_num: int, chunk_idx: int):
    if chunk_num <= 1:
        return items
    if chunk_idx < 0 or chunk_idx >= chunk_num:
        raise ValueError(f"chunk_idx={chunk_idx} must satisfy 0 <= chunk_idx < chunk_num={chunk_num}")
    chunk_size = (len(items) + chunk_num - 1) // chunk_num
    start = chunk_idx * chunk_size
    end = min(len(items), (chunk_idx + 1) * chunk_size)
    return items[start:end]


def _get_first_env_int(keys, default=None):
    for key in keys:
        value = os.environ.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return default


def _resolve_dp_shard():
    chunk_num = _get_first_env_int(
        ["WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS"],
        default=1,
    )
    chunk_idx = _get_first_env_int(
        ["RANK", "OMPI_COMM_WORLD_RANK", "SLURM_PROCID"],
        default=0,
    )
    return max(1, chunk_num), max(0, chunk_idx)


def _evaluate_main_disease_from_scores(scores_list, meta_list, candidates):
    labels = [candidates.index(meta["label"]) for meta in meta_list]
    y_true = np.array(labels)
    y_score = np.array(scores_list)
    try:
        auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.5

    print("Task: MIMIC-IV-CDM Main Disease Diagnoses")
    print(f"Macro AUC: {auc:.4f}")
    return pd.DataFrame([{"Task": "MIMIC-IV-CDM Main Disease Diagnoses", "Count": len(scores_list), "Metric": "Macro AUC", "Value": auc}])


def _evaluate_main_disease_logprobs(outputs, meta_list, tokenizer):
    candidates = _get_main_disease_candidates()
    candidate_ids = {}
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        token_ids_space = tokenizer.encode(" " + candidate, add_special_tokens=False)
        candidate_ids[candidate] = list(set([token_ids[0], token_ids_space[0]]))

    scores_list = []
    for index, output in enumerate(outputs):
        if not output.outputs:
            scores_list.append([0.0] * len(candidates))
            continue

        top_logprobs = output.outputs[0].logprobs[0]
        scores = []
        for candidate in candidates:
            score = -100.0
            for token_id in candidate_ids[candidate]:
                if token_id in top_logprobs:
                    score = max(score, top_logprobs[token_id].logprob)
            scores.append(score)

        scores = np.array(scores)
        exps = np.exp(scores)
        normalized_probs = exps / (np.sum(exps) + 1e-12)
        scores_list.append(normalized_probs)

    return _evaluate_main_disease_from_scores(scores_list, meta_list, candidates)


def _evaluate_main_disease_prompt_logprobs(llm, prompts, meta_list, tokenizer, lora_request=None):
    candidates = _get_main_disease_candidates()
    scores_list = []
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=None,
    )

    for prompt in prompts:
        base_token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        scored_prompts = []
        scored_token_ids = []
        for candidate in candidates:
            full_prompt = prompt + " " + candidate
            full_token_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
            scored_prompts.append(full_prompt)
            scored_token_ids.append(full_token_ids)

        outputs = llm.generate(scored_prompts, sampling_params, lora_request=lora_request)

        scores = []
        for output, token_ids in zip(outputs, scored_token_ids):
            prompt_logprobs = output.prompt_logprobs or []
            score = 0.0
            missing_token = False
            for position in range(len(base_token_ids), len(token_ids)):
                if position >= len(prompt_logprobs):
                    missing_token = True
                    break
                position_logprobs = prompt_logprobs[position]
                if not position_logprobs:
                    missing_token = True
                    break
                token_id = token_ids[position]
                if token_id not in position_logprobs:
                    missing_token = True
                    break
                score += float(position_logprobs[token_id].logprob)
            if missing_token:
                score = -100.0
            scores.append(score)

        scores = np.array(scores)
        exps = np.exp(scores - np.max(scores))
        normalized_probs = exps / (np.sum(exps) + 1e-12)
        scores_list.append(normalized_probs)

    return _evaluate_main_disease_from_scores(scores_list, meta_list, candidates)


def _evaluate_icd_code(outputs, meta_list):
    f1_scores = []
    for index, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip() if output.outputs else ""
        pred_set = {item.strip() for item in re.split(r"[\n,]", generated_text) if item.strip()}
        true_set = {item.strip() for item in str(meta_list[index]["label"]).split("\n") if item.strip()}

        tp = len(pred_set.intersection(true_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        if tp == 0:
            f1 = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

        if index < 3:
            print(f"\n=== Sample {index} ===")
            print(f"Generated: '{generated_text}'")
            print(f"True Codes: {true_set}")
            print(f"Pred Codes: {pred_set}")
            print(f"F1: {f1:.4f}")

    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    print(f"Task: MIMIC-IV-CDM ICD Code Diagnoses")
    print(f"Average F1 Score: {avg_f1:.4f}")
    return pd.DataFrame([{"Task": "MIMIC-IV-CDM ICD Code Diagnoses", "Count": len(outputs), "Metric": "Average F1", "Value": avg_f1}])


def main():
    parser = HfArgumentParser(MIMICIVCDMTestArguments)
    (script_args,) = parser.parse_args_into_dataclasses()

    print("=" * 80)
    print("MIMIC-IV-CDM LLM Test")
    print("=" * 80)
    if script_args.score_mode not in {"logprobs", "prompt_logprobs"}:
        raise ValueError(f"Unsupported score_mode: {script_args.score_mode}")
    if script_args.parallel_mode not in {"dp", "tp", "pp"}:
        raise ValueError(f"Unsupported parallel_mode: {script_args.parallel_mode}")
    print(f"Model path: {script_args.model_path}")
    print(f"Task: {script_args.task_name}")
    print(f"Table mode: {script_args.table_mode}")
    print(f"Max seq len: {script_args.max_seq_len}")
    print(f"Use sequence classification: {script_args.use_sequence_classification}")

    dataset = _load_test_dataset(script_args)

    if script_args.use_sequence_classification:
        if script_args.task_name != "MIMIC-IV-CDM Main Disease Diagnoses":
            raise NotImplementedError(
                "Sequence classification mode currently supports only 'MIMIC-IV-CDM Main Disease Diagnoses'."
            )
        candidates, label_to_id, id_to_label = _build_label_metadata(script_args.task_name)
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
                gpu_memory_utilization=0.9,
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
        print(f"Parallel mode: {script_args.parallel_mode}")
        print(f"Score mode: {script_args.score_mode}")
        vllm_model_path, tokenizer_path, lora_request = resolve_vllm_model_and_lora(
            script_args.model_path,
            script_args.base_model_path,
        )
        tokenizer = load_tokenizer(tokenizer_path)
        tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        prompts, meta_list = build_prompts_from_dataset(
            dataset,
            tokenizer,
            system_prompt="",
            max_seq_length=script_args.max_seq_len,
        )
        if script_args.parallel_mode == "dp":
            chunk_num, chunk_idx = _resolve_dp_shard()
            print(f"DP shard: {chunk_idx}/{chunk_num}")
            prompts = _slice_chunk(prompts, chunk_num, chunk_idx)
            meta_list = _slice_chunk(meta_list, chunk_num, chunk_idx)
        num_gpus = max(1, len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES") else 0)
        if num_gpus <= 0:
            try:
                import torch
                num_gpus = max(1, torch.cuda.device_count())
            except Exception:
                num_gpus = 1
        tp_size = script_args.tp_size if script_args.tp_size > 0 else num_gpus
        pp_size = script_args.pp_size
        if script_args.parallel_mode == "pp":
            if pp_size <= 1:
                pp_size = num_gpus
            if script_args.tp_size <= 0:
                tp_size = 1
        elif script_args.parallel_mode == "dp":
            tp_size = 1
            pp_size = 1
        if tp_size * pp_size > num_gpus:
            raise ValueError(
                f"Invalid parallel setup: tp_size={tp_size}, pp_size={pp_size}, visible_gpus={num_gpus}"
            )
        print(f"Using vLLM with tp_size={tp_size}, pp_size={pp_size}")
        llm = LLM(
            model=vllm_model_path,
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            trust_remote_code=True,
            max_model_len=script_args.max_seq_len,
            gpu_memory_utilization=0.9,
            enable_lora=lora_request is not None,
        )

        if script_args.task_name != "MIMIC-IV-CDM Main Disease Diagnoses":
            sampling_params = SamplingParams(temperature=0, max_tokens=64, logprobs=None)
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            metrics_df = _evaluate_icd_code(outputs, meta_list)
        else:
            if script_args.score_mode == "prompt_logprobs":
                metrics_df = _evaluate_main_disease_prompt_logprobs(
                    llm,
                    prompts,
                    meta_list,
                    tokenizer,
                    lora_request=lora_request,
                )
            else:
                sampling_params = SamplingParams(temperature=0, max_tokens=10, logprobs=20)
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
                metrics_df = _evaluate_main_disease_logprobs(outputs, meta_list, tokenizer)
        raw_df = None
        save_dir = script_args.output_dir

    save_dir = save_dir or os.path.join(script_args.model_path, "eval_logs")
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    if raw_df is not None:
        raw_path = os.path.join(save_dir, "raw_predictions.csv")
        raw_df.to_csv(raw_path, index=False)
        print(f"Raw predictions saved to {raw_path}")
    print(f"Results saved to {metrics_path}")


if __name__ == "__main__":
    main()
