import argparse
import importlib
import json
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

from ethos.constants import SpecialToken as ST
from ethos.vocabulary import Vocabulary


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


def _fit_stats(dataset, empty_token, max_samples, num_workers, num_buckets, min_numeric, max_text):
    nums, texts, raw_counts = defaultdict(list), defaultdict(Counter), Counter()
    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for df in tqdm(_iter_samples(dataset, empty_token, max_samples, num_workers), total=total, desc="Fit ETHOS vocab"):
        raw_counts.update(df["code"])

        numeric = pd.to_numeric(df["numeric_value"], errors="coerce")
        for code, values in df[numeric.notna()].assign(numeric_value=numeric[numeric.notna()].astype(float)).groupby("code")["numeric_value"]:
            nums[str(code)].extend(values.tolist())

        text = df["text_value"].fillna("").astype(str).str.strip()
        for code, values in df[text != ""].assign(text_value=text[text != ""]).groupby("code")["text_value"]:
            texts[str(code)].update(_norm_text(v) for v in values.tolist())

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
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite_output:
        for path in output_dir.glob("*"):
            path.unlink()

    static_roots = [x.strip().upper() for x in static_prefixes.split(",") if x.strip()]
    quantiles, text_values, raw_counts = _fit_stats(
        dataset,
        empty_context_token,
        max_samples,
        num_workers,
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
        },
        (output_dir / "build_metadata.json").open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    print(f"Subjects used: {len(dataset) if max_samples is None else min(len(dataset), max_samples)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Artifacts written to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_kwargs", default="{}")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument("--num_numeric_buckets", type=int, default=10)
    parser.add_argument("--min_numeric_values_per_code", type=int, default=20)
    parser.add_argument("--max_text_values_per_code", type=int, default=200)
    parser.add_argument("--static_prefixes", default="GENDER,RACE,ETHNICITY,MARITAL")
    parser.add_argument("--unknown_event_token", default="UNKNOWN_EVENT")
    parser.add_argument("--empty_context_token", default="NO_EVENT_CONTEXT")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    module_name, target_name = args.dataset.split(":")
    dataset_fn = getattr(importlib.import_module(module_name), target_name)
    dataset_kwargs = json.loads(args.dataset_kwargs)
    dataset = (
        _ConcatDataset([dataset_fn(**kwargs) for kwargs in dataset_kwargs])
        if isinstance(dataset_kwargs, list)
        else dataset_fn(**dataset_kwargs)
    )

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
    )


if __name__ == "__main__":
    main()
