import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.renji.renji_dataset import RenjiDataset
from ethos.constants import SpecialToken as ST
from ethos.vocabulary import Vocabulary


EHRSHOT_TASKS = [
    "guo_los",
    "guo_readmission",
    "guo_icu",
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hyponatremia",
    "lab_hypoglycemia",
    "lab_thrombocytopenia",
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
]
EICU_TASKS = [
    "mortality",
    "long_term_mortality",
    "readmission",
    "los_3day",
    "los_7day",
    "creatinine",
    "bilirubin",
    "platelets",
    "wbc",
    "final_acuity",
    "imminent_discharge",
]
EHR_BENCH_TASKS = [
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
    "ED_Hospitalization",
    "ED_Inpatient_Mortality",
    "ED_ICU_Tranfer_12hour",
    "ED_Reattendance_3day",
    "ED_Critical_Outcomes",
]
TIME_INTERVALS = [
    ("5m-15m", 5 * 60),
    ("15m-45m", 15 * 60),
    ("45m-1h15m", 45 * 60),
    ("1h15m-2h", 75 * 60),
    ("2h-3h", 2 * 3600),
    ("3h-5h", 3 * 3600),
    ("5h-8h", 5 * 3600),
    ("8h-12h", 8 * 3600),
    ("12h-18h", 12 * 3600),
    ("18h-1d", 18 * 3600),
    ("1d-2d", 1 * 86400),
    ("2d-4d", 2 * 86400),
    ("4d-7d", 4 * 86400),
    ("7d-12d", 7 * 86400),
    ("12d-20d", 12 * 86400),
    ("20d-30d", 20 * 86400),
    ("30d-2mt", 30 * 86400),
    ("2mt-6mt", 60 * 86400),
    ("=6mt", 180 * 86400),
]
MEDS_COLUMNS = ["subject_id", "time", "code", "numeric_value", "text_value", "unit", "omop_table"]


class _ConcatDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.offsets = np.cumsum([0] + [len(dataset) for dataset in datasets])

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, idx):
        dataset_idx = int(np.searchsorted(self.offsets[1:], idx, side="right"))
        return self.datasets[dataset_idx][idx - int(self.offsets[dataset_idx])]


def _first(batch):
    return batch[0]


def _norm_text(value):
    text = "" if value is None else str(value).strip().upper()
    out, was_sep = [], False
    for ch in text:
        if ch.isalnum():
            out.append(ch)
            was_sep = False
        elif not was_sep:
            out.append("_")
            was_sep = True
    return "".join(out).strip("_") or "UNKNOWN"


def _as_meds_df(sample, subject_id, empty_token):
    df = sample.copy() if isinstance(sample, pd.DataFrame) else sample["meds_table"].copy()
    for col in MEDS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[MEDS_COLUMNS].copy()
    if len(df) == 0:
        df = pd.DataFrame([{
            "subject_id": subject_id,
            "time": "1970-01-01 00:00:00",
            "code": empty_token,
            "numeric_value": None,
            "text_value": "",
            "unit": "",
            "omop_table": "observation",
        }])
    df["subject_id"] = subject_id
    df["time"] = df["time"].fillna("").astype(str)
    df["code"] = df["code"].fillna("").astype(str)
    df["text_value"] = df["text_value"].fillna("").astype(str)
    return df


def _iter_samples(dataset, empty_token, max_samples, num_workers):
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    source = dataset if max_samples is None else Subset(dataset, range(n))
    loader = DataLoader(source, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=_first)
    for i, sample in enumerate(loader):
        yield _as_meds_df(sample, i + 1, empty_token)


def _accumulate_stats(df, nums, texts, raw_counts):
    raw_counts.update(df["code"])

    numeric = pd.to_numeric(df["numeric_value"], errors="coerce")
    numeric_rows = df[numeric.notna()].assign(numeric_value=numeric[numeric.notna()].astype(float))
    for code, values in numeric_rows.groupby("code")["numeric_value"]:
        nums[str(code)].extend(values.tolist())

    text = df["text_value"].fillna("").astype(str).str.strip()
    text_rows = df[text != ""].assign(text_value=text[text != ""])
    for code, values in text_rows.groupby("code")["text_value"]:
        texts[str(code)].update(_norm_text(v) for v in values.tolist())


def _collect_stats_for_indices(dataset, indices, empty_token):
    nums, texts, raw_counts = defaultdict(list), defaultdict(Counter), Counter()
    for idx in indices:
        df = _as_meds_df(dataset[idx], idx + 1, empty_token)
        _accumulate_stats(df, nums, texts, raw_counts)
    return {
        "nums": dict(nums),
        "texts": {code: dict(counter) for code, counter in texts.items()},
        "raw_counts": dict(raw_counts),
    }


_fit_stats_worker_context = None


def _init_fit_stats_worker(dataset, empty_token):
    global _fit_stats_worker_context
    _fit_stats_worker_context = {
        "dataset": dataset,
        "empty_token": empty_token,
    }


def _fit_stats_worker(task):
    start, end = task
    partial = _collect_stats_for_indices(
        _fit_stats_worker_context["dataset"],
        range(start, end),
        _fit_stats_worker_context["empty_token"],
    )
    partial["task"] = (start, end)
    return partial


def _merge_partial_stats(partial, nums, texts, raw_counts):
    raw_counts.update(Counter(partial["raw_counts"]))
    for code, values in partial["nums"].items():
        nums[code].extend(values)
    for code, counter in partial["texts"].items():
        texts[code].update(Counter(counter))


def _save_stats_checkpoint(checkpoint_path, completed_tasks, nums, texts, raw_counts, total, num_processes, process_chunk_size):
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "completed_tasks": sorted(completed_tasks),
                "nums": dict(nums),
                "texts": {code: dict(counter) for code, counter in texts.items()},
                "raw_counts": dict(raw_counts),
                "total": total,
                "num_processes": num_processes,
                "process_chunk_size": process_chunk_size,
            },
            f,
        )


def _load_stats_checkpoint(checkpoint_path, nums, texts, raw_counts):
    with Path(checkpoint_path).open("rb") as f:
        checkpoint = pickle.load(f)
    for code, values in checkpoint["nums"].items():
        nums[code].extend(values)
    for code, counter in checkpoint["texts"].items():
        texts[code].update(Counter(counter))
    raw_counts.update(Counter(checkpoint["raw_counts"]))
    return {tuple(task) for task in checkpoint["completed_tasks"]}


def _fit_stats(
    dataset,
    empty_token,
    max_samples,
    num_workers,
    num_processes,
    process_chunk_size,
    checkpoint_path,
    checkpoint_every_chunks,
    num_buckets,
    min_numeric,
    max_text,
):
    nums, texts, raw_counts = defaultdict(list), defaultdict(Counter), Counter()
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if total == 0:
        return {}, {}, raw_counts

    if num_processes > 1:
        tasks = [(start, min(total, start + process_chunk_size)) for start in range(0, total, process_chunk_size)]
        completed_tasks = set()
        if checkpoint_path and Path(checkpoint_path).exists():
            completed_tasks = _load_stats_checkpoint(checkpoint_path, nums, texts, raw_counts)
            print(f"Loaded checkpoint: {checkpoint_path} ({len(completed_tasks)} chunks completed)")
        tasks = [task for task in tasks if task not in completed_tasks]
        merged_since_checkpoint = 0
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=num_processes,
            initializer=_init_fit_stats_worker,
            initargs=(dataset, empty_token),
        ) as pool:
            for partial in tqdm(
                pool.imap_unordered(_fit_stats_worker, tasks),
                total=len(tasks),
                desc=f"Fit ETHOS vocab ({num_processes} processes)",
            ):
                _merge_partial_stats(partial, nums, texts, raw_counts)
                completed_tasks.add(tuple(partial["task"]))
                merged_since_checkpoint += 1
                if checkpoint_path and merged_since_checkpoint >= checkpoint_every_chunks:
                    _save_stats_checkpoint(
                        checkpoint_path,
                        completed_tasks,
                        nums,
                        texts,
                        raw_counts,
                        total,
                        num_processes,
                        process_chunk_size,
                    )
                    merged_since_checkpoint = 0
        if checkpoint_path:
            _save_stats_checkpoint(
                checkpoint_path,
                completed_tasks,
                nums,
                texts,
                raw_counts,
                total,
                num_processes,
                process_chunk_size,
            )
    else:
        for df in tqdm(_iter_samples(dataset, empty_token, max_samples, num_workers), total=total, desc="Fit ETHOS vocab"):
            _accumulate_stats(df, nums, texts, raw_counts)

    quantiles = {}
    for code, values in nums.items():
        if len(values) >= min_numeric:
            qs = np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, num_buckets + 1)[1:-1])
            quantiles[code] = sorted({float(v) for v in qs.tolist() if np.isfinite(v)})
    text_values = {code: [v for v, _ in c.most_common(max_text)] for code, c in texts.items()}
    return quantiles, text_values, raw_counts


def _interval_estimates():
    return {
        stat: {label: int(seconds * 1_000_000) for label, seconds in TIME_INTERVALS}
        for stat in ("min", "q1", "mean", "median", "q3", "max")
    }


def _time_token(delta_us):
    if delta_us <= 0:
        return None
    delta_seconds, chosen = delta_us / 1_000_000.0, TIME_INTERVALS[0][0]
    for label, seconds in TIME_INTERVALS:
        if delta_seconds >= seconds:
            chosen = label
    return chosen


def _event_token(row, quantiles, text_values):
    code = str(row["code"]).strip()
    numeric = pd.to_numeric(row.get("numeric_value"), errors="coerce")
    if pd.notna(numeric) and quantiles.get(code):
        bucket = np.searchsorted(np.asarray(quantiles[code], dtype=np.float64), float(numeric), side="right") + 1
        return f"{code}//Q{bucket}"
    text = str(row.get("text_value", "")).strip()
    if text:
        value = _norm_text(text)
        allowed = text_values.get(code)
        return f"{code}//VALUE//{value if allowed is None or value in allowed else 'OTHER'}"
    return code


def _to_time_us(series):
    return pd.Series(
        [None if pd.isna(ts) else int(ts.value // 1000) for ts in pd.to_datetime(series, errors="coerce")],
        index=series.index,
        dtype="object",
    )


def _tokenize(df, static_roots, quantiles, text_values, vocab, unknown_token, empty_token):
    df = df.copy()
    df["time_us"] = _to_time_us(df["time"])
    known = set(vocab.stoi) if vocab is not None else None
    rows, counts, static_data = [], Counter(), {}

    for subject_id, group in df.groupby("subject_id", sort=True):
        group = group.sort_values(["time_us", "code"], na_position="last")
        static = {root: {"code": [f"{root}//UNKNOWN"], "time": [0]} for root in static_roots}
        events = []

        for _, row in group.iterrows():
            token = _event_token(row, quantiles, text_values)
            root = token.split("//", 1)[0].upper()
            if root in static_roots:
                static[root] = {"code": [token], "time": [0]}
                continue
            event_time = row["time_us"]
            event_time = 0 if event_time is None and not events else events[-1][0] if event_time is None else int(event_time)
            events.append((event_time, unknown_token if known is not None and token not in known else token))

        events = events or [(0, empty_token if known is None or empty_token in known else unknown_token)]
        timeline, prev_time = [], None
        for event_time, token in events:
            if prev_time is not None:
                gap = _time_token(event_time - prev_time)
                if gap:
                    timeline.append((event_time, unknown_token if known is not None and gap not in known else gap))
            timeline.append((event_time, token))
            prev_time = event_time

        end_token = ST.TIMELINE_END.value
        timeline.append((prev_time or 0, unknown_token if known is not None and end_token not in known else end_token))
        for root, value in static.items():
            token = value["code"][0]
            if known is not None and token not in known:
                fallback = f"{root}//UNKNOWN"
                static[root] = {"code": [fallback if fallback in known else unknown_token], "time": [0]}
        static_data[int(subject_id)] = static

        for event_time, token in timeline:
            rows.append({"subject_id": int(subject_id), "time": int(event_time), "code": token})
            counts[token] += 1

    return pd.DataFrame(rows).sort_values(["subject_id", "time", "code"]).reset_index(drop=True), counts, static_data


def _make_vocab(static_roots, raw_counts, quantiles, text_values, intervals, unknown_token, empty_token):
    tokens = [f"{root}//UNKNOWN" for root in static_roots]
    tokens += [label for label, _ in TIME_INTERVALS]
    tokens += [ST.TIMELINE_END.value, unknown_token, empty_token]
    for code, _ in raw_counts.most_common():
        tokens.append(code)
        tokens += [f"{code}//Q{i}" for i in range(1, len(quantiles.get(code, [])) + 2)]
        tokens += [f"{code}//VALUE//{value}" for value in text_values.get(code, [])]
        if code in text_values:
            tokens.append(f"{code}//VALUE//OTHER")
    return Vocabulary(list(dict.fromkeys(tokens)), intervals)


def build_from_meds_dataset(
    dataset,
    *,
    output_dir,
    overwrite_output=False,
    num_numeric_buckets=10,
    min_numeric_values_per_code=20,
    max_text_values_per_code=200,
    static_prefixes="GENDER,RACE,ETHNICITY,MARITAL",
    unknown_event_token="UNKNOWN_EVENT",
    empty_context_token="NO_EVENT_CONTEXT",
    max_samples=None,
    num_workers=0,
    num_processes=1,
    process_chunk_size=1000,
    checkpoint_path=None,
    checkpoint_every_chunks=5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path_obj = Path(checkpoint_path).resolve() if checkpoint_path else None
    if overwrite_output:
        for path in output_dir.glob("*"):
            if checkpoint_path_obj and path.resolve() == checkpoint_path_obj:
                continue
            path.unlink()

    static_roots = [x.strip().upper() for x in static_prefixes.split(",") if x.strip()]
    quantiles, text_values, raw_counts = _fit_stats(
        dataset,
        empty_context_token,
        max_samples,
        num_workers,
        num_processes,
        process_chunk_size,
        checkpoint_path,
        checkpoint_every_chunks,
        num_numeric_buckets,
        min_numeric_values_per_code,
        max_text_values_per_code,
    )
    intervals = _interval_estimates()
    vocab = _make_vocab(static_roots, raw_counts, quantiles, text_values, intervals, unknown_event_token, empty_context_token)
    vocab.dump(output_dir)

    with (output_dir / "raw_code_counts.csv").open("w", encoding="utf-8") as f:
        f.write("code,count\n")
        for code, count in raw_counts.most_common():
            f.write(f"{code},{count}\n")
    json.dump(quantiles, (output_dir / "quantiles.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(text_values, (output_dir / "text_values.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(intervals, (output_dir / "interval_estimates.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(
        {
            "input_source": "meds_dataset",
            "output_dir": str(output_dir),
            "num_subjects": len(dataset) if max_samples is None else min(len(dataset), max_samples),
            "vocab_size": len(vocab),
            "static_roots": static_roots,
            "num_workers": num_workers,
            "num_processes": num_processes,
            "process_chunk_size": process_chunk_size,
            "checkpoint_path": checkpoint_path,
            "checkpoint_every_chunks": checkpoint_every_chunks,
        },
        (output_dir / "build_metadata.json").open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    print(f"Subjects used: {len(dataset) if max_samples is None else min(len(dataset), max_samples)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Artifacts written to: {output_dir}")


def build_dataset(args):
    if args.dataset_name == "ehrshot":
        return _ConcatDataset([
            EHRSHOTDataset(
                root_dir=args.ehrshot_root_dir,
                sample_info_path=f"{args.ehrshot_root_dir}/index/ehrshot_train.csv",
                task_name=task_name,
                lazy_mode=True,
                table_mode="table_only",
                return_meds=True,
            )
            for task_name in EHRSHOT_TASKS
        ])

    if args.dataset_name == "eicu":
        return _ConcatDataset([
            EICUDataset(
                root_dir=args.eicu_root_dir,
                processed_dir=args.eicu_processed_dir,
                sample_info_path=f"{args.eicu_processed_dir}/sample_info_train.json",
                task_name=task_name,
                lazy_mode=True,
                shuffle=False,
                table_mode="table_only",
                return_meds=True,
            )
            for task_name in EICU_TASKS
        ])

    if args.dataset_name == "ehr_bench":
        os.environ["MIMIC_SKIP_SAMPLE_CACHE_CHECK"] = "1"
        return _ConcatDataset([
            MIMICIV(
                root_dir=args.ehr_bench_data_dir,
                sample_info_path=f"{args.ehr_bench_data_dir}/task_index/train/{task_name}.csv",
                lazy_mode=True,
                shuffle=False,
                table_mode="table_only",
                itemid_representation="code",
                return_meds=True,
            )
            for task_name in EHR_BENCH_TASKS
        ])

    if args.dataset_name == "mimic_iv_cdm":
        return MIMICIVCDM(
            root_dir=args.mimic_iv_cdm_root_dir,
            split="train",
            lazy_mode=True,
            shuffle=False,
            table_mode="table_only",
            task_name="MIMIC-IV-CDM Main Disease Diagnoses",
            return_meds=True,
            concept_map_dir=args.mimic_iv_cdm_concept_map_dir,
        )

    if args.dataset_name == "renji":
        return RenjiDataset(
            root_dir=args.renji_root_dir,
            split="train",
            table_mode="text_only",
            target_prediction_points=["day0", "day30", "day180", "day365"],
            shuffle=False,
            return_meds=True,
        )

    raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True, choices=["ehrshot", "eicu", "ehr_bench", "mimic_iv_cdm", "renji"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--ehrshot_root_dir", default="/data/EHR_data_public/EHRSHOT")
    parser.add_argument("--eicu_root_dir", default="/data/EHR_data_public/eicu-crd/2.0")
    parser.add_argument("--eicu_processed_dir", default="/data/zikun_workspace/eicu-crd/processed")
    parser.add_argument("--ehr_bench_data_dir", default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    parser.add_argument("--mimic_iv_cdm_root_dir", default="/data/EHR_data_public/mimic-iv-cdm")
    parser.add_argument("--mimic_iv_cdm_concept_map_dir", default="/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS")
    parser.add_argument("--renji_root_dir", default="/data/EHR_data_public/Renji")
    parser.add_argument("--num_numeric_buckets", type=int, default=10)
    parser.add_argument("--min_numeric_values_per_code", type=int, default=20)
    parser.add_argument("--max_text_values_per_code", type=int, default=200)
    parser.add_argument("--static_prefixes", default="GENDER,RACE,ETHNICITY,MARITAL")
    parser.add_argument("--unknown_event_token", default="UNKNOWN_EVENT")
    parser.add_argument("--empty_context_token", default="NO_EVENT_CONTEXT")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--process_chunk_size", type=int, default=1000)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--checkpoint_every_chunks", type=int, default=5)
    args = parser.parse_args()

    dataset = build_dataset(args)

    build_from_meds_dataset(
        dataset,
        output_dir=args.output_dir,
        overwrite_output=args.overwrite_output,
        num_numeric_buckets=args.num_numeric_buckets,
        min_numeric_values_per_code=args.min_numeric_values_per_code,
        max_text_values_per_code=args.max_text_values_per_code,
        static_prefixes=args.static_prefixes,
        unknown_event_token=args.unknown_event_token,
        empty_context_token=args.empty_context_token,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        num_processes=args.num_processes,
        process_chunk_size=args.process_chunk_size,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every_chunks=args.checkpoint_every_chunks,
    )


if __name__ == "__main__":
    main()
