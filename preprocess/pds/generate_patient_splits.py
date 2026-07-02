import argparse
import csv
import json
import os
import random
from collections import defaultdict


TASK_LABEL_FILES = {
    "severe_outcome": "severe_outcome.csv",
    "adverse_event_next_visit": "adverse_event_next_visit.csv",
}


def normalize_patient_id(value):
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def parse_csv_list(value):
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def discover_trials(root_dir):
    return sorted(
        name
        for name in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, name, "labels"))
    )


def read_patient_label_counts(label_path):
    patient_label_counts = defaultdict(lambda: {0: 0, 1: 0})
    with open(label_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_id = normalize_patient_id(row["patient_id"])
            patient_label_counts[patient_id][int(row["label"])] += 1
    return dict(patient_label_counts)


def split_trial_patients(patient_label_counts, seed, ratios):
    patient_ids = sorted(patient_label_counts)
    n_total = len(patient_ids)
    if n_total == 0:
        raise ValueError("Cannot split an empty patient set.")

    total_label_counts = {
        0: sum(counts[0] for counts in patient_label_counts.values()),
        1: sum(counts[1] for counts in patient_label_counts.values()),
    }
    if total_label_counts[0] == 0 or total_label_counts[1] == 0:
        raise ValueError("Patient split requires both positive and negative labels.")

    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    patient_ids.sort(
        key=lambda patient_id: (
            sum(patient_label_counts[patient_id].values()),
            abs(patient_label_counts[patient_id][1] - patient_label_counts[patient_id][0]),
        ),
        reverse=True,
    )

    n_train = int(n_total * ratios[0])
    n_val = int(n_total * ratios[1])
    target_patient_counts = {
        "train": n_train,
        "val": n_val,
        "test": n_total - n_train - n_val,
    }
    target_label_counts = {
        split: {
            0: total_label_counts[0] * target_patient_counts[split] / n_total,
            1: total_label_counts[1] * target_patient_counts[split] / n_total,
        }
        for split in ("train", "val", "test")
    }

    assigned_patients = {"train": [], "val": [], "test": []}
    assigned_label_counts = {
        "train": {0: 0, 1: 0},
        "val": {0: 0, 1: 0},
        "test": {0: 0, 1: 0},
    }

    for patient_id in patient_ids:
        patient_counts = patient_label_counts[patient_id]
        candidate_splits = [
            split
            for split in ("train", "val", "test")
            if len(assigned_patients[split]) < target_patient_counts[split]
        ]
        best_split = min(
            candidate_splits,
            key=lambda split: split_ratio_score(
                assigned_label_counts[split],
                patient_counts,
                target_label_counts[split],
            ),
        )
        assigned_patients[best_split].append(patient_id)
        assigned_label_counts[best_split][0] += patient_counts[0]
        assigned_label_counts[best_split][1] += patient_counts[1]

    return {
        split: sorted(patient_ids)
        for split, patient_ids in assigned_patients.items()
    }, assigned_label_counts


def split_ratio_score(current_counts, patient_counts, target_counts):
    return sum(
        abs(current_counts[label] + patient_counts[label] - target_counts[label])
        / target_counts[label]
        for label in (0, 1)
    )


def main():
    parser = argparse.ArgumentParser(description="Generate PDS patient-level train/val/test splits.")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--trial_ids", default=None, help="Comma-separated trial IDs. Defaults to all trials.")
    parser.add_argument(
        "--tasks",
        default="severe_outcome,adverse_event_next_visit",
        help="Comma-separated task names.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trial_ids = parse_csv_list(args.trial_ids) or discover_trials(args.root_dir)
    tasks = parse_csv_list(args.tasks)
    ratios = (args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio)
    if ratios[0] <= 0 or ratios[1] <= 0 or ratios[2] <= 0:
        raise ValueError("train/val/test ratios must all be positive.")

    output = {}
    for task_name in tasks:
        label_file = TASK_LABEL_FILES[task_name]
        output[task_name] = {}
        for trial_id in trial_ids:
            label_path = os.path.join(args.root_dir, trial_id, "labels", label_file)
            if not os.path.exists(label_path):
                continue

            patient_label_counts = read_patient_label_counts(label_path)
            split_patients, split_label_counts = split_trial_patients(
                patient_label_counts,
                seed=f"{args.seed}:{task_name}:{trial_id}",
                ratios=ratios,
            )
            output[task_name][trial_id] = split_patients
            counts_text = " ".join(
                f"{split}=patients:{len(split_patients[split])},labels:{split_label_counts[split]}"
                for split in ("train", "val", "test")
            )
            print(f"{task_name} trial {trial_id}: {counts_text}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
