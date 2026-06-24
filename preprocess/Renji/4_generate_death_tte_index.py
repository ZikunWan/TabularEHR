#!/usr/bin/env python3
"""Generate patient-level Renji death time-to-event index files."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional


DEFAULT_ROOT_DIR = "/data/EHR_data_public/Renji"
DEFAULT_HORIZON_DAYS = 1825


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate patient-level death TTE index files for Renji."
    )
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--encoding", default="utf-8-sig")
    return parser.parse_args()


def read_csv(path: str, encoding: str) -> List[Dict[str, str]]:
    encodings = [encoding]
    for extra in ("utf-8-sig", "utf-8", "gb18030"):
        if extra not in encodings:
            encodings.append(extra)
    last_error = None
    for current_encoding in encodings:
        try:
            with open(path, newline="", encoding=current_encoding) as file:
                return list(csv.DictReader(file))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def parse_time(value) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"} or text == "/":
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def parse_float(value) -> Optional[float]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "死亡",
        "deceased",
        "dead",
    }


def unique_split_files(all_samples: Iterable[Dict[str, str]]):
    by_split = {}
    seen = {}
    for row in all_samples:
        split = str(row.get("split", "")).strip()
        file_name = str(row.get("file_name", "")).strip()
        if not split or not file_name:
            continue
        seen.setdefault(split, set())
        by_split.setdefault(split, [])
        if file_name in seen[split]:
            continue
        seen[split].add(file_name)
        by_split[split].append(file_name)
    return by_split


def build_patient_info_map(root_dir: str, encoding: str):
    rows = read_csv(
        os.path.join(root_dir, "患儿基本信息总表251023_含免疫事件.csv"),
        encoding,
    )
    return {os.path.splitext(row["file_name"])[0]: row for row in rows}


def read_followup_days(root_dir: str, file_name: str, encoding: str):
    path = os.path.join(
        root_dir,
        "follow_ups",
        file_name if file_name.endswith(".csv") else f"{file_name}.csv",
    )
    rows = read_csv(path, encoding)
    values = []
    for row in rows:
        day = parse_float(row.get("术后天数"))
        report_time = parse_time(row.get("报告日期"))
        if day is not None:
            values.append((day, report_time))
    values.sort(key=lambda item: item[0])
    return values


def make_death_tte_row(
    root_dir: str,
    file_name: str,
    split: str,
    patient_info: Dict[str, str],
    horizon_days: int,
    encoding: str,
):
    followup_days = read_followup_days(root_dir, file_name, encoding)
    if not followup_days:
        return None, "no_followup_days"

    first_report = next(
        ((day, report_time) for day, report_time in followup_days if report_time),
        None,
    )
    if first_report is None:
        return None, "no_report_date"

    surgery_date = first_report[1] - timedelta(days=first_report[0])
    death_day = None
    if is_truthy(patient_info.get("is_deceased")):
        death_date = parse_time(patient_info.get("date_of_death"))
        if death_date is None:
            return None, "missing_death_time"
        death_day = max((death_date - surgery_date).total_seconds() / 86400.0, 0.0)

    nonnegative_days = [day for day, _ in followup_days if day >= 0]
    if not nonnegative_days:
        return None, "no_postoperative_followup"
    prediction_day = float(min(nonnegative_days))
    last_followup_day = float(max(day for day, _ in followup_days))
    if death_day is not None and death_day <= prediction_day:
        return None, "death_before_prediction"
    if death_day is None and last_followup_day <= prediction_day:
        return None, "no_followup_after_prediction"

    stage_end_day = prediction_day + float(horizon_days)
    event_observed = death_day is not None and death_day <= stage_end_day
    observed_day = (
        death_day if event_observed else min(last_followup_day, stage_end_day)
    )
    if observed_day <= prediction_day:
        return None, "nonpositive_duration"

    fname_key = os.path.splitext(file_name)[0]
    return (
        {
            "file_name": file_name,
            "fname_key": fname_key,
            "split": split,
            "task": "death_survival",
            "stage_id": 0,
            "stage_start_day": f"{prediction_day:.6f}",
            "stage_end_day": f"{stage_end_day:.6f}",
            "prediction_day": f"{prediction_day:.6f}",
            "cutoff_day": f"{prediction_day:.6f}",
            "observed_day": f"{observed_day:.6f}",
            "time_to_event": f"{observed_day - prediction_day:.6f}",
            "event_observed": int(event_observed),
            "stage_end_horizon": f"{stage_end_day - prediction_day:.6f}",
            "num_bins": int(horizon_days),
            "time_unit": "day",
        },
        None,
    )


def write_index(path: str, rows: List[Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "file_name",
        "fname_key",
        "split",
        "task",
        "stage_id",
        "stage_start_day",
        "stage_end_day",
        "prediction_day",
        "cutoff_day",
        "observed_day",
        "time_to_event",
        "event_observed",
        "stage_end_horizon",
        "num_bins",
        "time_unit",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.root_dir, "index")
    patient_info_map = build_patient_info_map(args.root_dir, args.encoding)
    all_samples = read_csv(os.path.join(args.root_dir, "all_samples.csv"), args.encoding)
    split_files = unique_split_files(all_samples)

    for split, file_names in sorted(split_files.items()):
        rows = []
        skipped = {}
        for file_name in file_names:
            fname_key = os.path.splitext(file_name)[0]
            patient_info = patient_info_map.get(fname_key)
            if patient_info is None:
                skipped["missing_patient_info"] = skipped.get("missing_patient_info", 0) + 1
                continue
            row, reason = make_death_tte_row(
                args.root_dir,
                file_name,
                split,
                patient_info,
                args.horizon_days,
                args.encoding,
            )
            if row is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            rows.append(row)

        output_path = os.path.join(output_dir, f"death_tte_{split}.csv")
        write_index(output_path, rows)
        events = sum(int(row["event_observed"]) for row in rows)
        print(
            f"{split}: rows={len(rows)}, events={events}, "
            f"censored={len(rows) - events}, skipped={skipped}"
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
