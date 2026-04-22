import argparse
import os

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


DEFAULT_PATIENTS_PATH = "/home/ma-user/sfs_turbo/sai6/yangqian/tmp_input/mimic-iv-3.1/hosp/patients.csv.gz"
DEFAULT_TASK_INDEX_DIR = "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/all"
DEFAULT_OUTPUT_DIR = "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/patient_data"

RISK_TASKS = [
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

POSITIVE_LABELS = {"yes", "true", "1", "1.0", "y"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate patient train/val/test splits by risk-task multilabel stratification.")
    parser.add_argument("--patients_path", type=str, default=DEFAULT_PATIENTS_PATH)
    parser.add_argument("--task_index_dir", type=str, default=DEFAULT_TASK_INDEX_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def normalize_label(raw):
    if raw is None:
        return ""
    s = str(raw).strip()
    for _ in range(3):
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        else:
            break
    return s.lower()


def build_patient_multilabel(subject_ids_df, task_index_dir):
    data = subject_ids_df[["subject_id"]].copy()
    data["subject_id_str"] = data["subject_id"].astype(str)

    for task in RISK_TASKS:
        csv_path = os.path.join(task_index_dir, f"{task}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Risk task csv not found: {csv_path}")

        task_df = pd.read_csv(csv_path, usecols=["subject_id", "target"])
        task_df["subject_id"] = task_df["subject_id"].astype(str)
        pos_subjects = set(task_df.loc[task_df["target"].map(lambda x: normalize_label(x) in POSITIVE_LABELS), "subject_id"].unique())
        data[task] = data["subject_id_str"].isin(pos_subjects).astype(int)

        pos_cnt = int(data[task].sum())
        print(f"  {task}: positive patients={pos_cnt} ({(100.0 * pos_cnt / len(data)):.2f}%)")

    return data


def stratified_split(multilabel_df, train_ratio, val_ratio, test_ratio, random_seed):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_ratio + val_ratio + test_ratio must be 1.0, got {total}")

    y = multilabel_df[RISK_TASKS].to_numpy(dtype=int)
    all_indices = np.arange(len(multilabel_df))

    holdout_ratio = val_ratio + test_ratio
    if holdout_ratio <= 0 or holdout_ratio >= 1:
        raise ValueError(f"val_ratio + test_ratio must be in (0,1), got {holdout_ratio}")

    split_1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=holdout_ratio,
        random_state=random_seed,
    )
    train_idx, holdout_idx = next(split_1.split(all_indices, y))

    holdout_rel_indices = np.arange(len(holdout_idx))
    holdout_y = y[holdout_idx]
    test_in_holdout = test_ratio / holdout_ratio

    split_2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_in_holdout,
        random_state=random_seed,
    )
    val_rel_idx, test_rel_idx = next(split_2.split(holdout_rel_indices, holdout_y))

    val_idx = holdout_idx[val_rel_idx]
    test_idx = holdout_idx[test_rel_idx]

    return train_idx, val_idx, test_idx


def report_split_stats(multilabel_df, split_name, idx):
    split_df = multilabel_df.iloc[idx]
    print(f"{split_name}: {len(split_df)} patients")
    for task in RISK_TASKS:
        rate = split_df[task].mean() if len(split_df) > 0 else 0.0
        print(f"  {task}: {rate * 100:.2f}%")


def main():
    args = parse_args()

    print("=" * 60)
    print("EHR-Bench Patient Split Generation (Risk-Task Stratified)")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir: {args.output_dir}")

    if not os.path.exists(args.patients_path):
        raise FileNotFoundError(f"patients_path not found: {args.patients_path}")
    if not os.path.isdir(args.task_index_dir):
        raise FileNotFoundError(f"task_index_dir not found: {args.task_index_dir}")

    patients_df = pd.read_csv(args.patients_path)
    if "subject_id" not in patients_df.columns:
        raise ValueError(f"subject_id column not found in {args.patients_path}")

    subject_ids = patients_df[["subject_id"]].drop_duplicates().reset_index(drop=True)
    print(f"Loaded unique patients: {len(subject_ids)}")

    print("\nBuilding risk-task multilabel fingerprints by patient...")
    multilabel_df = build_patient_multilabel(subject_ids, args.task_index_dir)

    print("\nRunning multilabel-stratified split...")
    train_idx, val_idx, test_idx = stratified_split(
        multilabel_df=multilabel_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed,
    )

    train_ids = subject_ids.iloc[train_idx].reset_index(drop=True)
    val_ids = subject_ids.iloc[val_idx].reset_index(drop=True)
    test_ids = subject_ids.iloc[test_idx].reset_index(drop=True)

    print("\nSplit sizes:")
    print(f"  Train: {len(train_ids)} ({100.0 * len(train_ids) / len(subject_ids):.2f}%)")
    print(f"  Val:   {len(val_ids)} ({100.0 * len(val_ids) / len(subject_ids):.2f}%)")
    print(f"  Test:  {len(test_ids)} ({100.0 * len(test_ids) / len(subject_ids):.2f}%)")

    print("\nPer-task positive rates by split:")
    report_split_stats(multilabel_df, "Train", train_idx)
    report_split_stats(multilabel_df, "Val", val_idx)
    report_split_stats(multilabel_df, "Test", test_idx)

    print("\nSaving csv files...")
    to_save = {
        "patients.csv": subject_ids,
        "train.csv": train_ids,
        "val.csv": val_ids,
        "test.csv": test_ids,
    }
    for filename, frame in to_save.items():
        output_path = os.path.join(args.output_dir, filename)
        frame.to_csv(output_path, index=False)
        print(f"  saved {filename}: {len(frame)}")

    train_set = set(train_ids["subject_id"].tolist())
    val_set = set(val_ids["subject_id"].tolist())
    test_set = set(test_ids["subject_id"].tolist())
    no_overlap = (len(train_set & val_set) == 0) and (len(train_set & test_set) == 0) and (len(val_set & test_set) == 0)

    print("\nVerification:")
    print(f"  total patients: {len(subject_ids)}")
    print(f"  train+val+test: {len(train_ids) + len(val_ids) + len(test_ids)}")
    print(f"  no overlap: {no_overlap}")
    print("=" * 60)


if __name__ == "__main__":
    main()
