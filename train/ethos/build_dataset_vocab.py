import json
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import HfArgumentParser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ETHOS_SRC = os.path.dirname(os.path.abspath(__file__))
if ETHOS_SRC not in sys.path:
    sys.path.insert(0, ETHOS_SRC)

from ethos.constants import STATIC_DATA_FN, SpecialToken as ST
from ethos.datasets.base import TimelineDataset
from ethos.vocabulary import Vocabulary


TIME_INTERVAL_SPEC = [
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


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print(*args, **kwargs)


@dataclass
class BuildArguments:
    input_dir: str = field(
        default="/data/zikun_workspace/code/.cache/ethos_exports/ehr_bench/train",
        metadata={"help": "Directory containing raw MEDS parquet shards exported for ETHOS."},
    )
    output_dir: str = field(
        default="/data/zikun_workspace/code/.cache/ethos_tokenized/ehr_bench/train",
        metadata={"help": "Directory to save ETHOS-ready vocab artifacts and tensorized shards."},
    )
    overwrite_output: bool = field(
        default=False,
        metadata={"help": "Overwrite output_dir if it already contains files."},
    )
    reuse_artifacts_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional train artifact directory. When set, reuse quantiles/vocab/intervals "
                "instead of fitting a new vocabulary."
            )
        },
    )
    num_numeric_buckets: int = field(
        default=10,
        metadata={"help": "Number of quantile buckets for numeric codes."},
    )
    min_numeric_values_per_code: int = field(
        default=20,
        metadata={"help": "Minimum observations before fitting numeric quantiles for a code."},
    )
    max_text_values_per_code: int = field(
        default=200,
        metadata={"help": "Keep top-K normalized text categories per code during train vocab fitting."},
    )
    static_prefixes: str = field(
        default="GENDER,RACE,ETHNICITY,MARITAL",
        metadata={"help": "Comma-separated roots to keep in static_data.pickle instead of the timeline."},
    )
    unknown_event_token: str = field(
        default="UNKNOWN_EVENT",
        metadata={"help": "Fallback token when applying a train vocab to unseen events."},
    )
    empty_context_token: str = field(
        default="NO_EVENT_CONTEXT",
        metadata={"help": "Fallback token when a timeline has no dynamic events after filtering."},
    )


def _list_parquet_files(input_dir: str) -> List[Path]:
    files = sorted(Path(input_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {input_dir}")
    return files


def _ensure_output_dir(output_dir: str, overwrite: bool):
    path = Path(output_dir)
    if path.exists():
        existing_files = list(path.glob("*"))
        if existing_files and not overwrite:
            raise FileExistsError(
                f"Output directory already contains files: {path}. "
                "Pass --overwrite_output True to overwrite."
            )
    path.mkdir(parents=True, exist_ok=True)


def _parse_static_prefixes(static_prefixes: str) -> List[str]:
    parsed = [item.strip().upper() for item in str(static_prefixes).split(",") if item and item.strip()]
    if not parsed:
        raise ValueError("static_prefixes cannot be empty")
    return parsed


def _normalize_text_fragment(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    normalized = []
    last_is_underscore = False
    for ch in text:
        keep = ch.isalnum()
        if keep:
            normalized.append(ch)
            last_is_underscore = False
            continue
        if not last_is_underscore:
            normalized.append("_")
            last_is_underscore = True
    text = "".join(normalized).strip("_")
    return text or "UNKNOWN"


def _fit_quantile_breaks(values: Sequence[float], num_buckets: int) -> List[float]:
    if not values:
        return []
    quantiles = np.linspace(0, 1, int(num_buckets) + 1)[1:-1]
    breaks = np.quantile(np.asarray(values, dtype=np.float64), quantiles)
    unique_breaks = sorted({float(v) for v in breaks.tolist() if np.isfinite(v)})
    return unique_breaks


def _build_interval_estimates() -> Dict[str, Dict[str, int]]:
    stats = {}
    for stat_name in ("min", "q1", "mean", "median", "q3", "max"):
        stats[stat_name] = {}
        for label, lower_bound_seconds in TIME_INTERVAL_SPEC:
            stats[stat_name][label] = int(lower_bound_seconds * 1_000_000)
    return stats


def _bucketize_time_gap(delta_us: int) -> Optional[str]:
    if delta_us <= 0:
        return None
    delta_seconds = delta_us / 1_000_000.0
    chosen = None
    for label, lower_bound_seconds in TIME_INTERVAL_SPEC:
        if delta_seconds >= lower_bound_seconds:
            chosen = label
    if chosen is None:
        return TIME_INTERVAL_SPEC[0][0]
    return chosen


def _load_artifacts_from_dir(artifact_dir: str):
    artifact_dir = Path(artifact_dir)
    quantiles_path = artifact_dir / "quantiles.json"
    intervals_path = artifact_dir / "interval_estimates.json"
    if not quantiles_path.is_file():
        raise FileNotFoundError(f"quantiles.json not found in {artifact_dir}")
    if not intervals_path.is_file():
        raise FileNotFoundError(f"interval_estimates.json not found in {artifact_dir}")

    vocab = Vocabulary.from_path(artifact_dir)
    with quantiles_path.open("r", encoding="utf-8") as f:
        quantiles = json.load(f)
    with intervals_path.open("r", encoding="utf-8") as f:
        intervals = json.load(f)
    return vocab, quantiles, intervals


def _fit_code_statistics(
    parquet_files: Sequence[Path],
    *,
    num_numeric_buckets: int,
    min_numeric_values_per_code: int,
    max_text_values_per_code: int,
) -> Tuple[Dict[str, List[float]], Dict[str, List[str]], Counter]:
    numeric_values_by_code: Dict[str, List[float]] = defaultdict(list)
    text_counter_by_code: Dict[str, Counter] = defaultdict(Counter)
    raw_code_counter: Counter = Counter()

    pbar = tqdm(
        parquet_files,
        desc="Fitting raw code statistics",
        disable=int(os.environ.get("LOCAL_RANK", "-1")) not in (-1, 0),
    )
    for parquet_path in pbar:
        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            continue
        raw_code_counter.update(df["code"].fillna("").astype(str).tolist())

        numeric_series = pd.to_numeric(df["numeric_value"], errors="coerce")
        numeric_mask = numeric_series.notna()
        if numeric_mask.any():
            numeric_df = df.loc[numeric_mask, ["code"]].copy()
            numeric_df["numeric_value"] = numeric_series.loc[numeric_mask].astype(float).tolist()
            for code, values in numeric_df.groupby("code")["numeric_value"]:
                numeric_values_by_code[str(code)].extend(values.tolist())

        text_series = df["text_value"].fillna("").astype(str).str.strip()
        text_mask = text_series != ""
        if text_mask.any():
            text_df = df.loc[text_mask, ["code"]].copy()
            text_df["text_value"] = text_series.loc[text_mask].tolist()
            for code, values in text_df.groupby("code")["text_value"]:
                normalized_values = [_normalize_text_fragment(value) for value in values.tolist()]
                text_counter_by_code[str(code)].update(normalized_values)

    quantiles_by_code: Dict[str, List[float]] = {}
    for code, values in numeric_values_by_code.items():
        if len(values) < int(min_numeric_values_per_code):
            continue
        quantiles_by_code[code] = _fit_quantile_breaks(values, int(num_numeric_buckets))

    allowed_text_values_by_code: Dict[str, List[str]] = {}
    for code, counter in text_counter_by_code.items():
        allowed = [value for value, _ in counter.most_common(int(max_text_values_per_code))]
        if allowed:
            allowed_text_values_by_code[code] = allowed

    return quantiles_by_code, allowed_text_values_by_code, raw_code_counter


def _event_to_token(
    row: pd.Series,
    *,
    quantiles_by_code: Dict[str, List[float]],
    allowed_text_values_by_code: Dict[str, List[str]],
) -> str:
    code = str(row.get("code", "")).strip()
    if not code:
        return ""

    numeric_value = pd.to_numeric(row.get("numeric_value"), errors="coerce")
    if pd.notna(numeric_value) and code in quantiles_by_code and len(quantiles_by_code[code]) > 0:
        bucket_idx = int(np.searchsorted(np.asarray(quantiles_by_code[code], dtype=np.float64), float(numeric_value), side="right")) + 1
        return f"{code}//Q{bucket_idx}"

    text_value = str(row.get("text_value", "")).strip()
    if text_value:
        normalized = _normalize_text_fragment(text_value)
        allowed = allowed_text_values_by_code.get(code)
        if allowed is None or normalized in allowed:
            return f"{code}//VALUE//{normalized}"
        return f"{code}//VALUE//OTHER"

    return code


def _parse_time_to_us(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    out = []
    for ts in timestamps:
        if pd.isna(ts):
            out.append(None)
        else:
            out.append(int(ts.value // 1000))
    return pd.Series(out, index=series.index, dtype="object")


def _copy_artifact_files(src_dir: str, output_dir: str):
    src = Path(src_dir)
    out = Path(output_dir)
    for name in ("quantiles.json", "interval_estimates.json"):
        shutil.copy2(src / name, out / name)
    vocab_file = next(src.glob("vocab_t*.csv"))
    shutil.copy2(vocab_file, out / vocab_file.name)


def _build_static_data_entry(static_code: str) -> dict:
    return {"code": [static_code], "time": [0]}


def _process_single_parquet(
    parquet_path: Path,
    *,
    output_dir: Path,
    static_roots: Sequence[str],
    quantiles_by_code: Dict[str, List[float]],
    allowed_text_values_by_code: Dict[str, List[str]],
    reuse_vocab: Optional[Vocabulary],
    unknown_event_token: str,
    empty_context_token: str,
) -> Tuple[Counter, Dict[int, dict]]:
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        processed_df = pd.DataFrame(columns=["subject_id", "time", "code"])
        processed_path = output_dir / parquet_path.name
        processed_df.to_parquet(processed_path, index=False)
        return Counter(), {}

    df = df.copy()
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="raise").astype(int)
    df["time_us"] = _parse_time_to_us(df["time"])
    df["code"] = df["code"].fillna("").astype(str)
    df["text_value"] = df["text_value"].fillna("").astype(str)

    token_counter: Counter = Counter()
    static_data: Dict[int, dict] = {}
    processed_rows: List[dict] = []
    known_tokens = set(reuse_vocab.stoi.keys()) if reuse_vocab is not None else None

    grouped = df.groupby("subject_id", sort=True)
    for subject_id, group in grouped:
        group = group.sort_values(["time_us", "code"], na_position="last").reset_index(drop=True)
        static_entry = {root: _build_static_data_entry(f"{root}//UNKNOWN") for root in static_roots}

        dynamic_events: List[Tuple[int, str]] = []
        for _, row in group.iterrows():
            token = _event_to_token(
                row,
                quantiles_by_code=quantiles_by_code,
                allowed_text_values_by_code=allowed_text_values_by_code,
            )
            if not token:
                continue

            root = token.split("//", 1)[0].upper()
            if root in static_roots:
                static_entry[root] = _build_static_data_entry(token)
                continue

            event_time = row.get("time_us")
            if event_time is None:
                event_time = 0 if not dynamic_events else dynamic_events[-1][0]
            event_time = int(event_time)

            if known_tokens is not None and token not in known_tokens:
                token = unknown_event_token
            dynamic_events.append((event_time, token))

        if not dynamic_events:
            fallback_token = empty_context_token
            if known_tokens is not None and fallback_token not in known_tokens:
                fallback_token = unknown_event_token
            dynamic_events = [(0, fallback_token)]

        final_events: List[Tuple[int, str]] = []
        prev_time = None
        for event_time, token in dynamic_events:
            if prev_time is not None:
                interval_token = _bucketize_time_gap(event_time - prev_time)
                if interval_token:
                    if known_tokens is None or interval_token in known_tokens:
                        final_events.append((event_time, interval_token))
                    else:
                        final_events.append((event_time, unknown_event_token))
            final_events.append((event_time, token))
            prev_time = event_time

        timeline_end_token = str(ST.TIMELINE_END)
        if known_tokens is not None and timeline_end_token not in known_tokens:
            timeline_end_token = unknown_event_token
        final_events.append((prev_time if prev_time is not None else 0, timeline_end_token))

        for root, item in static_entry.items():
            token = item["code"][0]
            if known_tokens is not None and token not in known_tokens:
                fallback = f"{root}//UNKNOWN"
                if fallback in known_tokens:
                    static_entry[root] = _build_static_data_entry(fallback)
                else:
                    static_entry[root] = _build_static_data_entry(unknown_event_token)

        static_data[int(subject_id)] = static_entry

        for event_time, token in final_events:
            processed_rows.append(
                {
                    "subject_id": int(subject_id),
                    "time": int(event_time),
                    "code": token,
                }
            )
            token_counter[token] += 1

    processed_df = pd.DataFrame(processed_rows, columns=["subject_id", "time", "code"])
    processed_df = processed_df.sort_values(["subject_id", "time", "code"]).reset_index(drop=True)
    processed_path = output_dir / parquet_path.name
    processed_df.to_parquet(processed_path, index=False)
    return token_counter, static_data


def main():
    parser = HfArgumentParser((BuildArguments,))
    args, = parser.parse_args_into_dataclasses()

    parquet_files = _list_parquet_files(args.input_dir)
    _ensure_output_dir(args.output_dir, overwrite=args.overwrite_output)
    output_dir = Path(args.output_dir)
    static_roots = _parse_static_prefixes(args.static_prefixes)

    is_reuse_mode = bool(args.reuse_artifacts_dir)
    if is_reuse_mode:
        base_vocab, quantiles_by_code, interval_estimates = _load_artifacts_from_dir(args.reuse_artifacts_dir)
        allowed_text_values_by_code: Dict[str, List[str]] = {}
        _copy_artifact_files(args.reuse_artifacts_dir, args.output_dir)
        rank0_print(f"Reuse mode enabled. Loaded artifacts from {args.reuse_artifacts_dir}")
    else:
        quantiles_by_code, allowed_text_values_by_code, raw_code_counter = _fit_code_statistics(
            parquet_files,
            num_numeric_buckets=args.num_numeric_buckets,
            min_numeric_values_per_code=args.min_numeric_values_per_code,
            max_text_values_per_code=args.max_text_values_per_code,
        )
        interval_estimates = _build_interval_estimates()
        base_vocab = None
        with (output_dir / "raw_code_counts.csv").open("w", encoding="utf-8") as f:
            f.write("code,count\n")
            for code, count in raw_code_counter.most_common():
                f.write(f"{code},{count}\n")
        with (output_dir / "quantiles.json").open("w", encoding="utf-8") as f:
            json.dump(quantiles_by_code, f, ensure_ascii=False, indent=2)
        with (output_dir / "interval_estimates.json").open("w", encoding="utf-8") as f:
            json.dump(interval_estimates, f, ensure_ascii=False, indent=2)

    rank0_print("=" * 88)
    rank0_print("Build ETHOS Dataset Vocabulary")
    rank0_print("=" * 88)
    rank0_print(f"Input dir: {args.input_dir}")
    rank0_print(f"Output dir: {args.output_dir}")
    rank0_print(f"Reuse artifacts dir: {args.reuse_artifacts_dir or '(fit new)'}")
    rank0_print(f"Static roots: {', '.join(static_roots)}")

    aggregate_token_counter: Counter = Counter()
    aggregate_static_data: Dict[int, dict] = {}

    pbar = tqdm(
        parquet_files,
        desc="Processing raw MEDS shards",
        disable=int(os.environ.get("LOCAL_RANK", "-1")) not in (-1, 0),
    )
    for parquet_path in pbar:
        token_counter, static_data = _process_single_parquet(
            parquet_path,
            output_dir=output_dir,
            static_roots=static_roots,
            quantiles_by_code=quantiles_by_code,
            allowed_text_values_by_code=allowed_text_values_by_code,
            reuse_vocab=base_vocab,
            unknown_event_token=args.unknown_event_token,
            empty_context_token=args.empty_context_token,
        )
        aggregate_token_counter.update(token_counter)
        aggregate_static_data.update(static_data)

    with (output_dir / STATIC_DATA_FN).open("wb") as f:
        pickle.dump(aggregate_static_data, f)

    if is_reuse_mode:
        vocab = base_vocab
    else:
        vocab_tokens: List[str] = []
        vocab_tokens.extend([f"{root}//UNKNOWN" for root in static_roots])
        for entry in aggregate_static_data.values():
            for item in entry.values():
                vocab_tokens.extend(item["code"])
        vocab_tokens.extend([label for label, _ in TIME_INTERVAL_SPEC])
        vocab_tokens.extend(
            [
                str(ST.TIMELINE_END),
                args.unknown_event_token,
                args.empty_context_token,
            ]
        )
        for token, _ in aggregate_token_counter.most_common():
            vocab_tokens.append(token)

        deduped_vocab = []
        seen = set()
        for token in vocab_tokens:
            if token not in seen:
                deduped_vocab.append(token)
                seen.add(token)

        vocab = Vocabulary(deduped_vocab, interval_estimates)
        vocab.dump(output_dir)

    if is_reuse_mode and not (output_dir / "interval_estimates.json").exists():
        with (output_dir / "interval_estimates.json").open("w", encoding="utf-8") as f:
            json.dump(interval_estimates, f, ensure_ascii=False, indent=2)
    if is_reuse_mode and not (output_dir / "quantiles.json").exists():
        with (output_dir / "quantiles.json").open("w", encoding="utf-8") as f:
            json.dump(quantiles_by_code, f, ensure_ascii=False, indent=2)

    with (output_dir / "token_counts.csv").open("w", encoding="utf-8") as f:
        f.write("token,count\n")
        for token, count in aggregate_token_counter.most_common():
            f.write(f"{token},{count}\n")

    processed_parquet_files = sorted(output_dir.glob("*.parquet"))
    if not processed_parquet_files:
        raise FileNotFoundError(f"No processed parquet files were written to {output_dir}")

    tensorize_pbar = tqdm(
        processed_parquet_files,
        desc="Tensorizing shards",
        disable=int(os.environ.get("LOCAL_RANK", "-1")) not in (-1, 0),
    )
    for parquet_path in tensorize_pbar:
        TimelineDataset.tensorize(parquet_path, output_dir / parquet_path.name, vocab)

    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "reuse_artifacts_dir": args.reuse_artifacts_dir,
        "num_numeric_buckets": int(args.num_numeric_buckets),
        "min_numeric_values_per_code": int(args.min_numeric_values_per_code),
        "max_text_values_per_code": int(args.max_text_values_per_code),
        "static_roots": static_roots,
        "unknown_event_token": args.unknown_event_token,
        "empty_context_token": args.empty_context_token,
        "num_subjects": int(len(aggregate_static_data)),
        "vocab_size": int(len(vocab)),
    }
    with (output_dir / "build_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    rank0_print(f"Subjects exported: {len(aggregate_static_data)}")
    rank0_print(f"Vocabulary size: {len(vocab)}")
    rank0_print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
