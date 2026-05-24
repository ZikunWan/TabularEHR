import json
import math
import multiprocessing as mp
import os
import random
import sys
from glob import glob
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.renji.renji_dataset import RenjiDataset
from hf_ehr.config import (
    CategoricalTCE,
    NumericalRangeTCE,
    load_tokenizer_config_and_metadata_from_path,
    save_tokenizer_config_to_path,
)
from hf_ehr.data.tokenization import CLMBRTokenizer
from train.Llama.train_ehrshot_llama import _load_clmbr_tokenizer


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print(*args, **kwargs)


class Reservoir:
    """Reservoir sampling buffer to cap numeric memory per code."""

    def __init__(self, max_size: int, rng: random.Random):
        self.max_size = max_size
        self.rng = rng
        self.values: List[float] = []
        self.total_seen: int = 0

    def add(self, value: float):
        self.total_seen += 1
        if self.max_size <= 0:
            return
        if len(self.values) < self.max_size:
            self.values.append(value)
            return
        idx = self.rng.randint(0, self.total_seen - 1)
        if idx < self.max_size:
            self.values[idx] = value


@dataclass
class ExpansionArguments:
    output_tokenizer_config_path: str = field(
        metadata={"help": "Explicit output tokenizer_config.json path."},
    )
    dataset_name: str = field(
        default="eicu",
        metadata={"help": "One of: ehrshot, eicu, ehr_bench, mimic_iv_cdm, renji."},
    )
    model_name_or_path: str = field(
        default="/data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr",
        metadata={"help": "Base model path (must resolve tokenizer_config.json)."},
    )
    overwrite_output: bool = field(
        default=False,
        metadata={"help": "Overwrite output_tokenizer_config_path if it already exists."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed."})
    add_categorical_entries: bool = field(
        default=True,
        metadata={"help": "Add CategoricalTCE entries for observed text values."},
    )
    add_numerical_entries: bool = field(
        default=True,
        metadata={"help": "Add NumericalRangeTCE entries with quantile bucketing for observed numeric values."},
    )
    num_numeric_buckets: int = field(
        default=10,
        metadata={"help": "Target number of quantile buckets per numeric code."},
    )
    min_numeric_values_per_code: int = field(
        default=20,
        metadata={"help": "Minimum numeric observations required before adding buckets for a code."},
    )
    skip_codes_with_existing_numerical_ranges: bool = field(
        default=True,
        metadata={"help": "Skip bucket generation for codes that already have numerical_range entries in base config."},
    )
    max_numeric_values_per_code: int = field(
        default=200000,
        metadata={"help": "Reservoir cap for numeric values per code to control memory."},
    )
    max_categorical_values_per_code: int = field(
        default=200,
        metadata={"help": "Keep only top-K categorical values per code (by frequency)."},
    )
    parse_numeric_strings: bool = field(
        default=True,
        metadata={"help": "If true, parse numeric-looking strings into numeric values for bucketing."},
    )
    verify_with_expanded_tokenizer: bool = field(
        default=False,
        metadata={"help": "Re-scan all scanned train+val examples with the expanded tokenizer and report unresolved rate."},
    )
    scan_num_workers: int = field(
        default=1,
        metadata={
            "help": (
                "Number of worker processes for sharded scanning. "
                "Only used when dataset_name is ehr_bench or mimic_iv_cdm."
            )
        },
    )
    scan_chunk_size: int = field(
        default=2000,
        metadata={"help": "Chunk size (in samples) for parallel scan progress granularity."},
    )
    show_progress: bool = field(
        default=True,
        metadata={"help": "Whether to show tqdm progress bars."},
    )
    report_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional JSON report output path."},
    )


@dataclass
class DataArguments:
    # EHRSHOT
    ehrshot_root_dir: str = field(default="/data/EHR_data_public/EHRSHOT")
    ehrshot_train_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv")
    ehrshot_val_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv")

    # eICU
    eicu_root_dir: str = field(default="/data/EHR_data_public/eicu-crd/2.0")
    eicu_processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    eicu_train_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json")
    eicu_val_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json")

    # EHR-Bench (MIMIC tabular)
    ehr_bench_data_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    ehr_bench_train_sample_info_path: Optional[str] = field(default=None)
    ehr_bench_val_sample_info_path: Optional[str] = field(default=None)
    ehr_bench_task_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional comma-separated EHR-Bench task/file stems to scan "
                "(e.g., 'ED_Hospitalization,Readmission_30day'). "
                "Applies to both train/val when explicit *_sample_info_path is not set."
            )
        },
    )
    ehr_bench_itemid_representation: str = field(default="code")
    ehr_bench_concept_map_dir: Optional[str] = field(default=None)

    # MIMIC-IV-CDM
    mimic_iv_cdm_root_dir: str = field(default="/data/EHR_data_public/mimic-iv-cdm")
    mimic_iv_cdm_concept_map_dir: Optional[str] = field(default=None)

    # Renji
    renji_root_dir: str = field(default="/data/EHR_data_public/Renji")
    renji_eval_split: str = field(default="test")

    lazy_mode: bool = field(default=True)


def _resolve_sample_info_path(dataset_name: str, split: str, data_args: DataArguments) -> Optional[str]:
    if dataset_name == "ehrshot":
        if split == "train":
            return data_args.ehrshot_train_info_path
        if split in {"val", "validation"}:
            return data_args.ehrshot_val_info_path
        raise ValueError("EHRSHOT currently supports split=train/val for this script.")

    if dataset_name == "eicu":
        if split == "train":
            return data_args.eicu_train_info_path
        if split in {"val", "validation"}:
            return data_args.eicu_val_info_path
        raise ValueError("eICU currently supports split=train/val for this script.")

    raise ValueError(f"Unsupported dataset_name='{dataset_name}' for sample_info_path resolution.")


def _resolve_ehr_bench_sample_info_paths(split: str, data_args: DataArguments) -> List[str]:
    if split == "train":
        if data_args.ehr_bench_train_sample_info_path:
            return [data_args.ehr_bench_train_sample_info_path]
        pattern = os.path.join(data_args.ehr_bench_data_dir, "task_index", "train", "*.csv")
    elif split in {"val", "validation"}:
        if data_args.ehr_bench_val_sample_info_path:
            return [data_args.ehr_bench_val_sample_info_path]
        pattern = os.path.join(data_args.ehr_bench_data_dir, "task_index", "val", "*.csv")
    else:
        raise ValueError("EHR-Bench currently supports split=train/val for this script.")

    paths = sorted(glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No EHR-Bench task index files found: {pattern}")

    if data_args.ehr_bench_task_names:
        task_whitelist = {
            item.strip()
            for item in str(data_args.ehr_bench_task_names).split(",")
            if item and item.strip()
        }
        if not task_whitelist:
            raise ValueError("--ehr_bench_task_names was provided but no valid task names were parsed.")

        filtered_paths = []
        found_tasks = set()
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem in task_whitelist:
                filtered_paths.append(path)
                found_tasks.add(stem)

        missing_tasks = sorted(task_whitelist - found_tasks)
        if missing_tasks:
            raise FileNotFoundError(
                f"Requested EHR-Bench tasks not found in split '{split}': {missing_tasks}"
            )

        paths = filtered_paths

    return paths


def _configure_runtime_for_dataset(exp_args: ExpansionArguments):
    ds = exp_args.dataset_name.strip().lower()
    if ds == "ehr_bench":
        # Avoid nested multiprocessing explosion:
        # outer scan workers + MIMIC dataset internal workers should stay aligned.
        inner_workers = str(max(1, int(exp_args.scan_num_workers)))
        os.environ["MIMIC_TABLE_LENGTH_WORKERS"] = inner_workers
        os.environ["MIMIC_SAMPLE_CACHE_WORKERS"] = inner_workers
        rank0_print(
            "Configured MIMIC internal workers: "
            f"MIMIC_TABLE_LENGTH_WORKERS={inner_workers}, "
            f"MIMIC_SAMPLE_CACHE_WORKERS={inner_workers}"
        )


def _load_datasets_for_split(exp_args: ExpansionArguments, data_args: DataArguments, split: str):
    ds = exp_args.dataset_name.strip().lower()

    if ds == "ehrshot":
        sample_info_path = _resolve_sample_info_path(ds, split, data_args)
        dataset = EHRSHOTDataset(
            root_dir=data_args.ehrshot_root_dir,
            sample_info_path=sample_info_path,
            task_name=None,
            lazy_mode=data_args.lazy_mode,
            table_mode="table_only",
            max_samples=None,
            return_meds=True,
        )
        return [(split, dataset)]

    if ds == "eicu":
        sample_info_path = _resolve_sample_info_path(ds, split, data_args)
        dataset = EICUDataset(
            root_dir=data_args.eicu_root_dir,
            processed_dir=data_args.eicu_processed_dir,
            sample_info_path=sample_info_path,
            task_name=None,
            lazy_mode=data_args.lazy_mode,
            shuffle=False,
            table_mode="table_only",
            max_samples=None,
            return_meds=True,
        )
        return [(split, dataset)]

    if ds == "ehr_bench":
        sample_info_paths = _resolve_ehr_bench_sample_info_paths(split, data_args)
        datasets = []
        iterator = sample_info_paths
        if exp_args.show_progress:
            iterator = tqdm(sample_info_paths, desc=f"Loading datasets [{split}]")
        for sample_info_path in iterator:
            dataset = MIMICIV(
                root_dir=data_args.ehr_bench_data_dir,
                sample_info_path=sample_info_path,
                lazy_mode=data_args.lazy_mode,
                shuffle=False,
                table_mode="table_only",
                max_samples=None,
                itemid_representation=data_args.ehr_bench_itemid_representation,
                concept_map_dir=data_args.ehr_bench_concept_map_dir,
                return_meds=True,
            )
            source_name = os.path.splitext(os.path.basename(sample_info_path))[0]
            datasets.append((f"{split}/{source_name}", dataset))
        return datasets

    if ds == "mimic_iv_cdm":
        dataset = MIMICIVCDM(
            root_dir=data_args.mimic_iv_cdm_root_dir,
            split=split,
            lazy_mode=data_args.lazy_mode,
            shuffle=False,
            task_name="MIMIC-IV-CDM Main Disease Diagnoses",
            max_samples=None,
            return_meds=True,
            concept_map_dir=data_args.mimic_iv_cdm_concept_map_dir,
        )
        return [(split, dataset)]

    if ds == "renji":
        renji_split = data_args.renji_eval_split if split in {"val", "validation"} else split
        dataset = RenjiDataset(
            root_dir=data_args.renji_root_dir,
            split=renji_split,
            table_mode="text_only",
            target_prediction_points=["day0", "day30", "day180", "day365"],
            shuffle=False,
            return_meds=True,
        )
        dataset.samples = [sample for sample in dataset.samples if sample["metric"] == "all"]
        return [(renji_split, dataset)]

    raise ValueError(f"Unsupported dataset_name='{exp_args.dataset_name}'.")


def _load_train_val_datasets(exp_args: ExpansionArguments, data_args: DataArguments):
    split_datasets = []
    for split in ("train", "val"):
        split_datasets.extend(_load_datasets_for_split(exp_args, data_args, split=split))
    return split_datasets


def _event_attr(event, key: str):
    if isinstance(event, dict):
        return event[key]
    return getattr(event, key)


def _get_sample_hf_ehr_events(dataset_name: str, dataset, index: int):
    ds = dataset_name.strip().lower()

    if ds == "eicu":
        sample = dataset.sample_info[index]
        _, _, hf_ehr_events = dataset.meds_input_process(sample, return_hf_ehr_events=True)
        return hf_ehr_events

    if ds == "mimic_iv_cdm":
        index_item = dataset.list_data[index]
        category = index_item["category"]
        hadm_id = index_item["hadm_id"]
        cur_item = dataset.raw_data[category][hadm_id]
        _, _, hf_ehr_events = dataset.meds_input_process(cur_item, return_hf_ehr_events=True)
        return hf_ehr_events

    if ds == "renji":
        index_item = dataset.samples[index]
        df_followup = dataset._load_followup_data(index_item)
        first_row = df_followup.iloc[0]
        surgery_date = pd.to_datetime(first_row["报告日期"]) - pd.Timedelta(days=float(first_row["术后天数"]))
        static_features = dataset._get_static_features(index_item["fname_key"])
        _, _, hf_ehr_events = dataset.meds_input_process(
            subject_id=index_item["fname_key"],
            static_features=static_features,
            df_followup=df_followup,
            surgery_date=surgery_date,
        )
        return hf_ehr_events

    sample = dataset[index]
    return sample.get("hf_ehr_events") if isinstance(sample, dict) else None


def _try_parse_numeric(value, parse_numeric_strings: bool) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        v = float(value)
        return v if math.isfinite(v) else None

    if parse_numeric_strings and isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        numeric_chars = txt.lstrip("+-").replace(".", "", 1)
        if numeric_chars.isdigit():
            v = float(txt)
            return v if math.isfinite(v) else None
        return None

    return None


def _to_text_value(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        txt = value.strip()
        return txt if txt else None
    if isinstance(value, bool):
        return str(value)
    txt = str(value).strip()
    return txt if txt else None


def _quantile_ranges(values: List[float], num_buckets: int) -> List[Tuple[float, float]]:
    if len(values) == 0:
        return []
    if num_buckets < 1:
        return [(-float("inf"), float("inf"))]

    arr = np.asarray(values, dtype=np.float64)
    q = np.linspace(0.0, 1.0, num_buckets + 1)
    edges = np.quantile(arr, q)

    # Interior boundaries with de-duplication.
    boundaries: List[float] = []
    for x in edges[1:-1]:
        fx = float(x)
        if not math.isfinite(fx):
            continue
        if boundaries and fx <= boundaries[-1]:
            continue
        boundaries.append(fx)

    ranges: List[Tuple[float, float]] = []
    prev = -float("inf")
    for b in boundaries:
        ranges.append((prev, b))
        prev = b
    ranges.append((prev, float("inf")))
    return ranges


def _summarize_top(counter: Counter, k: int = 10) -> List[Tuple[str, int]]:
    return counter.most_common(k)


def _scan_dataset_events(
    dataset_name: str,
    dataset,
    indices,
    *,
    base_tokenizer: CLMBRTokenizer,
    parse_numeric_strings: bool,
    max_numeric_values_per_code: int,
    seed: int,
    progress_desc: Optional[str] = None,
    show_progress: bool = False,
):
    rng = random.Random(seed)
    code_counts: Counter = Counter()
    missing_code_counter: Counter = Counter()
    value_mismatch_counter: Counter = Counter()
    unresolved_total = 0
    total_events = 0
    categorical_counts: Dict[str, Counter] = defaultdict(Counter)
    numeric_reservoirs: Dict[str, Reservoir] = {}

    iterator = indices
    if show_progress:
        iterator = tqdm(
            indices,
            desc=progress_desc,
            disable=False,
        )

    for idx in iterator:
        events = _get_sample_hf_ehr_events(dataset_name, dataset, idx)
        if events is None:
            continue

        for event in events:
            code = _event_attr(event, "code")
            if code is None:
                continue
            code = str(code).strip()
            if not code:
                continue

            total_events += 1
            code_counts[code] += 1

            mapped = base_tokenizer.convert_event_to_token(event)
            if mapped is None:
                unresolved_total += 1
                if code in base_tokenizer.code_2_token:
                    value_mismatch_counter[code] += 1
                else:
                    missing_code_counter[code] += 1

            value = _event_attr(event, "value")
            numeric_value = _try_parse_numeric(value, parse_numeric_strings=parse_numeric_strings)
            if numeric_value is not None:
                reservoir = numeric_reservoirs.get(code)
                if reservoir is None:
                    reservoir = Reservoir(max_numeric_values_per_code, rng)
                    numeric_reservoirs[code] = reservoir
                reservoir.add(numeric_value)
                continue

            text_value = _to_text_value(value)
            if text_value is not None:
                categorical_counts[code][text_value] += 1

    return {
        "total_events": total_events,
        "unresolved_total": unresolved_total,
        "code_counts": code_counts,
        "missing_code_counter": missing_code_counter,
        "value_mismatch_counter": value_mismatch_counter,
        "categorical_counts": dict(categorical_counts),
        "numeric_values": {code: reservoir.values for code, reservoir in numeric_reservoirs.items()},
    }


def _merge_scan_result(
    result: dict,
    *,
    code_counts: Counter,
    missing_code_counter: Counter,
    value_mismatch_counter: Counter,
    categorical_counts: Dict[str, Counter],
    numeric_reservoirs: Dict[str, Reservoir],
    merge_rng: random.Random,
    max_numeric_values_per_code: int,
):
    code_counts.update(result["code_counts"])
    missing_code_counter.update(result["missing_code_counter"])
    value_mismatch_counter.update(result["value_mismatch_counter"])
    for code, counter in result["categorical_counts"].items():
        categorical_counts[code].update(counter)

    for code, values in result["numeric_values"].items():
        reservoir = numeric_reservoirs.get(code)
        if reservoir is None:
            reservoir = Reservoir(max_numeric_values_per_code, merge_rng)
            numeric_reservoirs[code] = reservoir
        for value in values:
            reservoir.add(value)

    return int(result["total_events"]), int(result["unresolved_total"])


_FORK_SCAN_CONTEXT = None


def _scan_worker_chunk(task):
    # NOTE: uses forked global context so dataset objects are inherited read-only.
    dataset_idx, start_idx, end_idx = task
    ctx = _FORK_SCAN_CONTEXT
    split_name, dataset = ctx["split_datasets"][dataset_idx]
    local_cap = max(1, int(ctx["worker_local_numeric_cap"]))
    indices = range(start_idx, end_idx)
    seed = ctx["seed"] + dataset_idx * 1000003 + start_idx * 7919
    result = _scan_dataset_events(
        ctx["dataset_name"],
        dataset,
        indices,
        base_tokenizer=ctx["base_tokenizer"],
        parse_numeric_strings=ctx["parse_numeric_strings"],
        max_numeric_values_per_code=local_cap,
        seed=seed,
        show_progress=False,
    )
    result["split_name"] = split_name
    result["start_idx"] = int(start_idx)
    result["end_idx"] = int(end_idx)
    return result


def main():
    parser = HfArgumentParser((ExpansionArguments, DataArguments))
    exp_args, data_args = parser.parse_args_into_dataclasses()
    output_tokenizer_config_path = exp_args.output_tokenizer_config_path

    if exp_args.num_numeric_buckets < 1:
        raise ValueError("--num_numeric_buckets must be >= 1")
    if exp_args.min_numeric_values_per_code < 1:
        raise ValueError("--min_numeric_values_per_code must be >= 1")
    if exp_args.max_numeric_values_per_code < 1:
        raise ValueError("--max_numeric_values_per_code must be >= 1")
    if exp_args.max_categorical_values_per_code < 1:
        raise ValueError("--max_categorical_values_per_code must be >= 1")
    if exp_args.scan_num_workers < 1:
        raise ValueError("--scan_num_workers must be >= 1")
    if exp_args.scan_chunk_size < 1:
        raise ValueError("--scan_chunk_size must be >= 1")

    if os.path.exists(output_tokenizer_config_path) and not exp_args.overwrite_output:
        raise FileExistsError(
            f"Output already exists: {output_tokenizer_config_path}. "
            "Pass --overwrite_output True to overwrite."
        )

    _configure_runtime_for_dataset(exp_args)

    # Resolve tokenizer config path strictly from model_name_or_path.
    base_tokenizer = _load_clmbr_tokenizer(exp_args.model_name_or_path)
    tokenizer_config_path = base_tokenizer.path_to_tokenizer_config

    split_datasets = _load_train_val_datasets(exp_args, data_args)
    split_sizes = {"train": 0, "val": 0}
    for split_name, ds in split_datasets:
        base_split = split_name.split("/", 1)[0]
        if base_split not in split_sizes:
            split_sizes[base_split] = 0
        split_sizes[base_split] += len(ds)
    total_samples_scanned = int(sum(split_sizes.values()))

    rank0_print("=" * 88)
    rank0_print("Build Dataset-Specific Tokenizer Config")
    rank0_print("=" * 88)
    rank0_print(f"Dataset: {exp_args.dataset_name}")
    rank0_print("Splits: train + val")
    rank0_print(f"Model path: {exp_args.model_name_or_path}")
    rank0_print(f"Tokenizer config: {tokenizer_config_path}")
    rank0_print(f"Output tokenizer config: {output_tokenizer_config_path}")
    rank0_print(f"Dataset size [train]: {split_sizes.get('train', 0)}")
    rank0_print(f"Dataset size [val]: {split_sizes.get('val', 0)}")
    rank0_print(f"Scanned samples (full): {total_samples_scanned}")
    rank0_print(f"Numeric buckets: {exp_args.num_numeric_buckets}")
    rank0_print(f"Min numeric observations/code: {exp_args.min_numeric_values_per_code}")
    rank0_print(f"Categorical expansion enabled: {exp_args.add_categorical_entries}")
    rank0_print(f"Numerical expansion enabled: {exp_args.add_numerical_entries}")
    rank0_print(f"Scan workers: {exp_args.scan_num_workers}")
    rank0_print(f"Scan chunk size: {exp_args.scan_chunk_size}")
    rank0_print(f"Show progress: {exp_args.show_progress}")

    rng = random.Random(exp_args.seed)
    code_counts: Counter = Counter()
    missing_code_counter: Counter = Counter()
    value_mismatch_counter: Counter = Counter()
    unresolved_total = 0
    total_events = 0

    categorical_counts: Dict[str, Counter] = defaultdict(Counter)
    numeric_reservoirs: Dict[str, Reservoir] = {}
    parallel_eligible = exp_args.dataset_name.strip().lower() in {"ehr_bench", "mimic_iv_cdm"}
    use_parallel_scan = parallel_eligible and exp_args.scan_num_workers > 1
    if use_parallel_scan:
        rank0_print("Parallel shard scan enabled.")
        global _FORK_SCAN_CONTEXT
        worker_local_numeric_cap = max(1, exp_args.max_numeric_values_per_code // exp_args.scan_num_workers)
        _FORK_SCAN_CONTEXT = {
            "dataset_name": exp_args.dataset_name,
            "split_datasets": split_datasets,
            "base_tokenizer": base_tokenizer,
            "parse_numeric_strings": exp_args.parse_numeric_strings,
            "max_numeric_values_per_code": exp_args.max_numeric_values_per_code,
            "worker_local_numeric_cap": worker_local_numeric_cap,
            "seed": exp_args.seed,
        }
        tasks = []
        for dataset_idx, (_, dataset) in enumerate(split_datasets):
            n = len(dataset)
            for start_idx in range(0, n, exp_args.scan_chunk_size):
                end_idx = min(n, start_idx + exp_args.scan_chunk_size)
                tasks.append((dataset_idx, start_idx, end_idx))

        mp_ctx = mp.get_context("fork")
        with mp_ctx.Pool(processes=exp_args.scan_num_workers) as pool:
            pbar = tqdm(
                pool.imap_unordered(_scan_worker_chunk, tasks),
                total=len(tasks),
                desc="Scanning chunks",
                disable=not exp_args.show_progress,
            )
            completed_chunks = 0
            completed_samples = 0
            for result in pbar:
                cur_total, cur_unresolved = _merge_scan_result(
                    result,
                    code_counts=code_counts,
                    missing_code_counter=missing_code_counter,
                    value_mismatch_counter=value_mismatch_counter,
                    categorical_counts=categorical_counts,
                    numeric_reservoirs=numeric_reservoirs,
                    merge_rng=rng,
                    max_numeric_values_per_code=exp_args.max_numeric_values_per_code,
                )
                total_events += cur_total
                unresolved_total += cur_unresolved
                completed_chunks += 1
                completed_samples += int(result["end_idx"]) - int(result["start_idx"])
                if exp_args.show_progress:
                    pbar.set_postfix(
                        chunks=f"{completed_chunks}/{len(tasks)}",
                        samples=f"{completed_samples}/{total_samples_scanned}",
                        unresolved=unresolved_total,
                    )
        _FORK_SCAN_CONTEXT = None
    else:
        if exp_args.scan_num_workers > 1 and not parallel_eligible:
            rank0_print("Scan workers > 1 requested, but current dataset uses serial scan path.")
        for dataset_idx, (split_name, dataset) in enumerate(split_datasets):
            result = _scan_dataset_events(
                exp_args.dataset_name,
                dataset,
                range(len(dataset)),
                base_tokenizer=base_tokenizer,
                parse_numeric_strings=exp_args.parse_numeric_strings,
                max_numeric_values_per_code=exp_args.max_numeric_values_per_code,
                seed=exp_args.seed + dataset_idx * 1000003,
                progress_desc=f"Scanning events [{split_name}]",
                show_progress=True,
            )
            cur_total, cur_unresolved = _merge_scan_result(
                result,
                code_counts=code_counts,
                missing_code_counter=missing_code_counter,
                value_mismatch_counter=value_mismatch_counter,
                categorical_counts=categorical_counts,
                numeric_reservoirs=numeric_reservoirs,
                merge_rng=rng,
                max_numeric_values_per_code=exp_args.max_numeric_values_per_code,
            )
            total_events += cur_total
            unresolved_total += cur_unresolved

    unresolved_rate = (100.0 * unresolved_total / total_events) if total_events > 0 else 0.0
    rank0_print(f"Total events scanned: {total_events}")
    rank0_print(f"Unresolved before expansion: {unresolved_total} ({unresolved_rate:.2f}%)")
    rank0_print(f"Missing-code events: {sum(missing_code_counter.values())}")
    rank0_print(f"Value-mismatch events: {sum(value_mismatch_counter.values())}")
    rank0_print(f"Top missing codes: {_summarize_top(missing_code_counter, 12)}")
    rank0_print(f"Top mismatch codes: {_summarize_top(value_mismatch_counter, 12)}")

    base_config, base_metadata = load_tokenizer_config_and_metadata_from_path(tokenizer_config_path)

    existing_numeric_codes = {entry.code for entry in base_config if entry.type == "numerical_range"}
    existing_categorical = defaultdict(set)
    for entry in base_config:
        if entry.type == "categorical":
            categories = tuple(entry.tokenization.get("categories", []))
            existing_categorical[entry.code].add(categories)

    expanded_config = list(base_config)

    added_categorical_entries = 0
    added_numeric_entries = 0
    skipped_numeric_existing = 0
    skipped_numeric_few_samples = 0

    if exp_args.add_categorical_entries:
        for code, counter in categorical_counts.items():
            if not counter:
                continue
            selected_values = counter.most_common(exp_args.max_categorical_values_per_code)
            for value, _ in selected_values:
                category_tuple = (value,)
                if category_tuple in existing_categorical[code]:
                    continue
                expanded_config.append(
                    CategoricalTCE(
                        code=code,
                        tokenization={"categories": [value]},
                    )
                )
                existing_categorical[code].add(category_tuple)
                added_categorical_entries += 1

    if exp_args.add_numerical_entries:
        for code, reservoir in numeric_reservoirs.items():
            values = reservoir.values
            if len(values) < exp_args.min_numeric_values_per_code:
                skipped_numeric_few_samples += 1
                continue

            if exp_args.skip_codes_with_existing_numerical_ranges and code in existing_numeric_codes:
                skipped_numeric_existing += 1
                continue

            ranges = _quantile_ranges(values, exp_args.num_numeric_buckets)
            for range_start, range_end in ranges:
                expanded_config.append(
                    NumericalRangeTCE(
                        code=code,
                        tokenization={
                            "unit": "None",
                            "range_start": range_start,
                            "range_end": range_end,
                        },
                    )
                )
                added_numeric_entries += 1

            existing_numeric_codes.add(code)

    Path(output_tokenizer_config_path).parent.mkdir(parents=True, exist_ok=True)

    expansion_meta = {
        "timestamp": datetime.now().isoformat(),
        "dataset_name": exp_args.dataset_name,
        "splits": ["train", "val"],
        "split_sizes": split_sizes,
        "source_tokenizer_config_path": tokenizer_config_path,
        "scanned_samples": total_samples_scanned,
        "dataset_size": total_samples_scanned,
        "scan_num_workers": exp_args.scan_num_workers,
        "parallel_scan_enabled": use_parallel_scan,
        "total_events": total_events,
        "unresolved_before": unresolved_total,
        "unresolved_before_rate": unresolved_rate,
        "added_code_entries": 0,
        "added_categorical_entries": added_categorical_entries,
        "added_numeric_entries": added_numeric_entries,
        "skipped_numeric_existing": skipped_numeric_existing,
        "skipped_numeric_few_samples": skipped_numeric_few_samples,
        "num_numeric_buckets": exp_args.num_numeric_buckets,
        "min_numeric_values_per_code": exp_args.min_numeric_values_per_code,
        "max_categorical_values_per_code": exp_args.max_categorical_values_per_code,
        "max_numeric_values_per_code": exp_args.max_numeric_values_per_code,
    }

    metadata = dict(base_metadata)
    metadata["auto_expansion"] = expansion_meta

    save_tokenizer_config_to_path(output_tokenizer_config_path, expanded_config, metadata=metadata)

    expanded_tokenizer = CLMBRTokenizer(path_to_tokenizer_config=output_tokenizer_config_path)

    rank0_print("-" * 88)
    rank0_print("Expansion Summary")
    rank0_print("-" * 88)
    rank0_print(f"Base token entries: {len(base_config)}")
    rank0_print(f"Expanded token entries: {len(expanded_config)}")
    rank0_print("Added code entries: 0")
    rank0_print(f"Added categorical entries: {added_categorical_entries}")
    rank0_print(f"Added numerical_range entries: {added_numeric_entries}")
    rank0_print(f"Skipped numeric codes (already had ranges): {skipped_numeric_existing}")
    rank0_print(f"Skipped numeric codes (few samples): {skipped_numeric_few_samples}")
    rank0_print(f"Base tokenizer vocab size: {base_tokenizer.vocab_size}")
    rank0_print(f"Expanded tokenizer vocab size: {expanded_tokenizer.vocab_size}")
    rank0_print(f"Saved expanded tokenizer config to: {output_tokenizer_config_path}")

    verify_unresolved_total = None
    verify_unresolved_rate = None
    if exp_args.verify_with_expanded_tokenizer:
        verify_unresolved_total = 0
        verify_total = 0
        for split_name, dataset in split_datasets:
            pbar = tqdm(
                range(len(dataset)),
                desc=f"Verifying expanded tokenizer [{split_name}]",
                disable=int(os.environ.get("LOCAL_RANK", "-1")) not in (-1, 0),
            )
            for idx in pbar:
                events = _get_sample_hf_ehr_events(exp_args.dataset_name, dataset, idx)
                if events is None:
                    continue
                for event in events:
                    verify_total += 1
                    if expanded_tokenizer.convert_event_to_token(event) is None:
                        verify_unresolved_total += 1
        verify_unresolved_rate = (100.0 * verify_unresolved_total / verify_total) if verify_total > 0 else 0.0
        rank0_print(
            f"Unresolved after expansion (verification): {verify_unresolved_total} ({verify_unresolved_rate:.2f}%)"
        )

    report = {
        "expansion": expansion_meta,
        "top_missing_codes_before": _summarize_top(missing_code_counter, 50),
        "top_mismatch_codes_before": _summarize_top(value_mismatch_counter, 50),
        "base_tokenizer_vocab_size": int(base_tokenizer.vocab_size),
        "expanded_tokenizer_vocab_size": int(expanded_tokenizer.vocab_size),
    }
    if verify_unresolved_total is not None:
        report["verify_unresolved_after"] = int(verify_unresolved_total)
        report["verify_unresolved_after_rate"] = float(verify_unresolved_rate)

    if exp_args.report_path:
        Path(exp_args.report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(exp_args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        rank0_print(f"Saved report to: {exp_args.report_path}")


if __name__ == "__main__":
    main()
