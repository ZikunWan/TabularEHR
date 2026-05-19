"""
Cache table-length statistics for eICU and EHRSHOT pretraining contexts.

The cached artifacts are meant to guide context-window choices before building
next_token_prediction and contrastive_learning sample_info files.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import json
import math
import os
from functools import partial

import pandas as pd
from tqdm import tqdm


def summarize(values):
    series = pd.Series(values, dtype="float64")
    if series.empty:
        return {"count": 0}
    quantiles = series.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "min": int(series.min()),
        "p50": float(quantiles[0.5]),
        "p75": float(quantiles[0.75]),
        "p90": float(quantiles[0.9]),
        "p95": float(quantiles[0.95]),
        "p99": float(quantiles[0.99]),
        "max": int(series.max()),
    }


def save_outputs(output_dir, prefix, patient_rows, window_rows, summary):
    os.makedirs(output_dir, exist_ok=True)
    patient_path = os.path.join(output_dir, f"{prefix}_patients.csv")
    window_path = os.path.join(output_dir, f"{prefix}_windows.csv")
    summary_path = os.path.join(output_dir, f"{prefix}_summary.json")
    pd.DataFrame(patient_rows).to_csv(patient_path, index=False)
    pd.DataFrame(window_rows).to_csv(window_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved patient lengths: {patient_path}")
    print(f"Saved window lengths: {window_path}")
    print(f"Saved summary: {summary_path}")


def load_eicu_split_map(processed_dir):
    stay_to_split = {}
    patient_to_split = {}
    for split in ["train", "val", "test"]:
        sample_path = os.path.join(processed_dir, f"sample_info_{split}.json")
        with open(sample_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        for sample in samples:
            stay_to_split[str(sample["icustay_id"])] = split
            patient_to_split[str(sample["patient_id"])] = split
    return stay_to_split, patient_to_split


def stable_split_from_patient_id(patient_id, train_ratio, val_ratio):
    value = int(hashlib.md5(str(patient_id).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def load_offsets(path, offset_column):
    if not os.path.exists(path):
        return pd.Series([], dtype="float64")
    df = pd.read_csv(path, usecols=[offset_column], low_memory=False)
    offsets = pd.to_numeric(df[offset_column], errors="coerce")
    return offsets[offsets >= 0]


def count_offsets(offsets, max_offset=None):
    if max_offset is None:
        return int(offsets.shape[0])
    return int((offsets <= max_offset).sum())


def load_eicu_patient_info(patient_dir):
    patient_path = os.path.join(patient_dir, "patient.csv")
    if not os.path.exists(patient_path):
        return None, 0
    patient_df = pd.read_csv(patient_path, low_memory=False)
    if patient_df.empty:
        return None, 0
    row = patient_df.iloc[0]
    patient_id = str(row["uniquepid"])
    static_count = 2
    ethnicity = str(row.get("ethnicity", "unknown"))
    if ethnicity and ethnicity != "unknown":
        static_count += 1
    return patient_id, static_count


def process_eicu_patient_dir(patient_dir, stay_to_split, patient_to_split, train_ratio, val_ratio):
    icustay_id = os.path.basename(patient_dir)
    patient_id, static_count = load_eicu_patient_info(patient_dir)
    lab_offsets = load_offsets(os.path.join(patient_dir, "lab.csv"), "labresultoffset")
    med_offsets = load_offsets(os.path.join(patient_dir, "medication.csv"), "drugstartoffset")
    infusion_offsets = load_offsets(os.path.join(patient_dir, "infusionDrug.csv"), "infusionoffset")
    lab_count = count_offsets(lab_offsets)
    med_count = count_offsets(med_offsets)
    infusion_count = count_offsets(infusion_offsets)
    max_values = [
        float(offsets.max())
        for offsets in [lab_offsets, med_offsets, infusion_offsets]
        if not offsets.empty
    ]
    max_offset = None if len(max_values) == 0 else max(max_values)
    full_dynamic_count = lab_count + med_count + infusion_count
    full_table_length = static_count + full_dynamic_count
    classification_split = stay_to_split.get(icustay_id, "unused")
    split = patient_to_split.get(
        patient_id,
        stable_split_from_patient_id(patient_id, train_ratio, val_ratio),
    )
    patient_row = {
        "dataset": "eicu",
        "icustay_id": icustay_id,
        "patient_id": patient_id,
        "split": split,
        "classification_split": classification_split,
        "static_rows": static_count,
        "lab_rows": lab_count,
        "med_rows": med_count,
        "infusion_rows": infusion_count,
        "dynamic_rows": full_dynamic_count,
        "table_length": full_table_length,
        "max_offset_minutes": max_offset,
        "max_offset_hours": None if max_offset is None else max_offset / 60.0,
    }

    obs_hours = 0 if max_offset is None else int(math.ceil(max_offset / 60.0))
    window_rows = [
        {
            "dataset": "eicu",
            "icustay_id": icustay_id,
            "split": split,
            "obs_hours": obs_hours,
            "context_begin": 0,
            "context_end": obs_hours * 60,
            "table_length": full_table_length,
            "dynamic_rows": full_dynamic_count,
        }
    ]
    return patient_row, window_rows


def cache_eicu(args):
    stay_to_split, patient_to_split = load_eicu_split_map(args.eicu_processed_dir)
    patient_dirs = sorted(glob.glob(os.path.join(args.eicu_processed_dir, "patients", "*")))
    patient_rows = []
    window_rows = []
    worker_fn = partial(
        process_eicu_patient_dir,
        stay_to_split=stay_to_split,
        patient_to_split=patient_to_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for patient_row, rows in tqdm(
            executor.map(worker_fn, patient_dirs, chunksize=args.chunksize),
            total=len(patient_dirs),
            desc="eICU table lengths",
        ):
            patient_rows.append(patient_row)
            window_rows.extend(rows)

    patient_df = pd.DataFrame(patient_rows)
    window_df = pd.DataFrame(window_rows)
    summary = {
        "dataset": "eicu",
        "num_patient_dirs": int(len(patient_rows)),
        "num_stays_in_classification_index": int((patient_df["classification_split"] != "unused").sum()),
        "num_unused_patient_dirs": int((patient_df["classification_split"] == "unused").sum()),
        "full_table_length": summarize(patient_df["table_length"]),
        "full_dynamic_rows": summarize(patient_df["dynamic_rows"]),
        "full_context_table_length": summarize(window_df["table_length"]),
        "num_windows": int(len(window_rows)),
        "split_counts": patient_df["split"].value_counts().to_dict(),
        "classification_split_counts": patient_df["classification_split"].value_counts().to_dict(),
    }
    save_outputs(args.output_dir, "eicu", patient_rows, window_rows, summary)


def load_ehrshot_split_map(root_dir):
    patient_to_split = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(root_dir, "index", f"ehrshot_{split}.csv")
        df = pd.read_csv(path, usecols=["patient_id"], low_memory=False)
        for patient_id in df["patient_id"].dropna().astype(str).unique():
            patient_to_split[patient_id] = split
    return patient_to_split


def process_ehrshot_patient_path(patient_path, split_map, window_rows_count, stride_rows, min_rows):
    patient_id = os.path.splitext(os.path.basename(patient_path))[0]
    split = split_map.get(patient_id, "unused")
    df = pd.read_csv(patient_path, usecols=["omop_table"], low_memory=False)
    person_rows = int((df["omop_table"] == "person").sum())
    non_person_indices = df.index[df["omop_table"] != "person"].tolist()
    non_person_rows = len(non_person_indices)
    patient_row = {
        "dataset": "ehrshot",
        "patient_id": patient_id,
        "split": split,
        "person_rows": person_rows,
        "non_person_rows": non_person_rows,
        "table_length": person_rows + non_person_rows,
    }
    window_rows = []
    for start_pos in range(0, non_person_rows, stride_rows):
        end_pos = min(start_pos + window_rows_count, non_person_rows)
        if end_pos - start_pos < min_rows:
            continue
        window_rows.append(
            {
                "dataset": "ehrshot",
                "patient_id": patient_id,
                "split": split,
                "context_begin": start_pos,
                "context_end": end_pos - 1,
                "period_begin": int(non_person_indices[start_pos]),
                "period_end": int(non_person_indices[end_pos - 1]),
                "table_length": person_rows + end_pos - start_pos,
                "non_person_rows": end_pos - start_pos,
                "person_rows": person_rows,
            }
        )
        if end_pos == non_person_rows:
            break
    return patient_row, window_rows


def cache_ehrshot(args):
    split_map = load_ehrshot_split_map(args.ehrshot_root_dir)
    patient_paths = sorted(glob.glob(os.path.join(args.ehrshot_root_dir, "patient_ehr", "*.csv")))
    patient_rows = []
    window_rows = []
    worker_fn = partial(
        process_ehrshot_patient_path,
        split_map=split_map,
        window_rows_count=args.ehrshot_window_rows,
        stride_rows=args.ehrshot_stride_rows,
        min_rows=args.ehrshot_min_rows,
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for patient_row, rows in tqdm(
            executor.map(worker_fn, patient_paths, chunksize=args.chunksize),
            total=len(patient_paths),
            desc="EHRSHOT table lengths",
        ):
            patient_rows.append(patient_row)
            window_rows.extend(rows)

    patient_df = pd.DataFrame(patient_rows)
    window_df = pd.DataFrame(window_rows)
    summary = {
        "dataset": "ehrshot",
        "num_patient_files": int(len(patient_rows)),
        "num_patients_in_classification_index": int((patient_df["split"] != "unused").sum()),
        "num_unused_patient_files": int((patient_df["split"] == "unused").sum()),
        "full_table_length": summarize(patient_df["table_length"]),
        "full_non_person_rows": summarize(patient_df["non_person_rows"]),
        "window_table_length": summarize(window_df["table_length"]),
        "num_windows": int(len(window_rows)),
        "split_counts": patient_df["split"].value_counts().to_dict(),
        "window_split_counts": window_df["split"].value_counts().to_dict(),
        "window_rows": int(args.ehrshot_window_rows),
        "stride_rows": int(args.ehrshot_stride_rows),
    }
    save_outputs(args.output_dir, "ehrshot", patient_rows, window_rows, summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["eicu", "ehrshot", "all"], default="all")
    parser.add_argument("--eicu_processed_dir", type=str, default="/data/zikun_workspace/eicu-crd/processed")
    parser.add_argument("--ehrshot_root_dir", type=str, default="/data/EHR_data_public/EHRSHOT")
    parser.add_argument("--output_dir", type=str, default="data/pretraining_table_lengths")
    parser.add_argument("--ehrshot_window_rows", type=int, default=2048)
    parser.add_argument("--ehrshot_stride_rows", type=int, default=2048)
    parser.add_argument("--ehrshot_min_rows", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.dataset in ["eicu", "all"]:
        cache_eicu(args)
    if args.dataset in ["ehrshot", "all"]:
        cache_ehrshot(args)


if __name__ == "__main__":
    main()
