from __future__ import annotations

import math
import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset.renji.renji_dataset import RenjiDataset, _is_main_process, _rank0_print


STAGE_SPECS = (
    {"stage_id": 0, "start_day": 0.0, "end_day": 31.0, "num_bins": 31},
    {"stage_id": 1, "start_day": 31.0, "end_day": 181.0, "num_bins": 150},
    {"stage_id": 2, "start_day": 181.0, "end_day": 366.0, "num_bins": 185},
)
MAX_SURVIVAL_BINS = 185
EVENT_LABEL_COLUMN = "他克莫司浓度_label"


def load_patient_subset(patient_subset_path):
    if patient_subset_path is None:
        return None
    with open(patient_subset_path, "r", encoding="utf-8") as file:
        patient_files = json.load(file)
    if not isinstance(patient_files, list):
        raise ValueError("patient_subset_path must contain a JSON list")
    return {
        os.path.splitext(os.path.basename(str(patient_file)))[0]
        for patient_file in patient_files
    }


def build_survival_instruction(task_schema, sample):
    stage_window = (
        f"[{int(sample['stage_start_day'])}, {int(sample['stage_end_day'])}) "
        "days post-transplant"
    )
    instruction = task_schema["instruction_template"].format(
        prediction_day=f"{sample['prediction_day']:g}",
        stage_window=stage_window,
    )
    return instruction, stage_window


def build_piecewise_survival_target(
    time_to_event: float,
    event_observed: bool,
    num_bins: int,
    max_bins: int = MAX_SURVIVAL_BINS,
):
    """Encode follow-up time using one-day intervals (k, k + 1]."""
    if time_to_event < 0:
        raise ValueError("time_to_event must be non-negative")
    if not 0 < num_bins <= max_bins:
        raise ValueError("num_bins must be in [1, max_bins]")

    observed_time = min(float(time_to_event), float(num_bins))
    bin_exposure = np.zeros(max_bins, dtype=np.float32)
    event_bins = np.zeros(max_bins, dtype=np.float32)
    stage_mask = np.zeros(max_bins, dtype=np.float32)
    stage_mask[:num_bins] = 1.0

    full_bins = min(int(math.floor(observed_time)), num_bins)
    if full_bins:
        bin_exposure[:full_bins] = 1.0
    if full_bins < num_bins:
        bin_exposure[full_bins] = observed_time - full_bins

    if event_observed and 0.0 < observed_time <= num_bins:
        event_bin = min(int(math.ceil(observed_time) - 1), num_bins - 1)
        event_bins[event_bin] = 1.0

    return bin_exposure, event_bins, stage_mask


def _parse_binary_label(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return 1 if float(value) > 0 else 0
    except (TypeError, ValueError):
        return None


class RenjiTacrolimusSurvivalDataset(RenjiDataset):
    """One time-to-event sample per patient and postoperative stage."""

    STAGE_SPECS = STAGE_SPECS
    MAX_SURVIVAL_BINS = MAX_SURVIVAL_BINS

    def __init__(
        self,
        root_dir,
        split="train",
        max_samples=None,
        table_mode="table_only",
        shuffle=False,
        patient_subset_path=None,
    ):
        self.patient_subset_path = patient_subset_path
        self.patient_subset = load_patient_subset(patient_subset_path)
        self.root_dir = root_dir
        self.split = split
        self.max_samples = max_samples
        if table_mode not in {
            "text_only",
            "table_only",
            "table_plus_rest_text",
        }:
            raise ValueError(f"Invalid table_mode: {table_mode}")
        self.table_mode = table_mode
        self.target_metrics = None
        self.target_prediction_points = []
        self.active_points = []
        self.shuffle = shuffle
        self.return_meds = False
        self.task_schema = deepcopy(self.TASK_INFO)
        self.followup_dir = os.path.join(self.root_dir, "follow_ups")
        self.index_dir = os.path.join(self.root_dir, "index")
        self._init_configs()
        self._load_auxiliary_data()

        split_table = pd.read_csv(
            os.path.join(self.root_dir, "all_samples.csv"),
            encoding="utf-8-sig",
        )
        split_files = split_table.loc[
            split_table["split"] == split,
            "file_name",
        ].drop_duplicates()
        self.filenames = split_files.astype(str).tolist()
        self._valid_followup_cache = {}
        self.samples = self._build_index()

    def _read_raw_followup(self, fname):
        path = os.path.join(
            self.followup_dir,
            fname if fname.endswith(".csv") else f"{fname}.csv",
        )
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "术后天数" not in df.columns:
            return pd.DataFrame()
        df["术后天数"] = pd.to_numeric(df["术后天数"], errors="coerce")
        if "报告日期" in df.columns:
            df["报告日期"] = pd.to_datetime(df["报告日期"], errors="coerce")
            df = df.sort_values(["术后天数", "报告日期"], na_position="last")
        else:
            df = df.sort_values("术后天数", na_position="last")
        return df.dropna(subset=["术后天数"]).reset_index(drop=True)

    def _build_index(self):
        _rank0_print(f"[{self.split}] Building tacrolimus survival sample index...")
        samples = []
        split_filenames = self.filenames
        if self.patient_subset is not None:
            split_filenames = [
                fname
                for fname in self.filenames
                if os.path.splitext(os.path.basename(str(fname)))[0]
                in self.patient_subset
            ]
            _rank0_print(
                f"[{self.split}] Patient subset filter: "
                f"{len(split_filenames)}/{len(self.filenames)} split patients retained "
                f"from {self.patient_subset_path}"
            )
            if not split_filenames:
                raise ValueError(
                    f"No {self.split} patients matched {self.patient_subset_path}"
                )

        for fname in tqdm(
            split_filenames,
            desc=f"[{self.split}] Survival indexing",
            disable=not _is_main_process(),
        ):
            fname_key = os.path.splitext(fname)[0]
            if fname_key not in self.patient_info_map:
                continue
            raw_followup = self._read_raw_followup(fname)

            for spec in self.STAGE_SPECS:
                stage_rows = raw_followup[
                    (raw_followup["术后天数"] >= spec["start_day"])
                    & (raw_followup["术后天数"] < spec["end_day"])
                ]
                if stage_rows.empty:
                    continue

                prediction_day = float(stage_rows["术后天数"].iloc[0])
                future_rows = stage_rows[stage_rows["术后天数"] > prediction_day]
                event_day = None
                if EVENT_LABEL_COLUMN in future_rows.columns:
                    for _, row in future_rows.iterrows():
                        if _parse_binary_label(row[EVENT_LABEL_COLUMN]) == 1:
                            event_day = float(row["术后天数"])
                            break

                event_observed = event_day is not None
                observed_day = (
                    event_day
                    if event_observed
                    else float(stage_rows["术后天数"].iloc[-1])
                )
                time_to_event = max(0.0, observed_day - prediction_day)
                exposure, event_bins, stage_mask = build_piecewise_survival_target(
                    time_to_event=time_to_event,
                    event_observed=event_observed,
                    num_bins=spec["num_bins"],
                )
                samples.append(
                    {
                        "fname": fname,
                        "fname_key": fname_key,
                        "stage_id": spec["stage_id"],
                        "stage_start_day": spec["start_day"],
                        "stage_end_day": spec["end_day"],
                        "num_bins": spec["num_bins"],
                        "prediction_day": prediction_day,
                        "cutoff_day": prediction_day,
                        "observed_day": observed_day,
                        "time_to_event": time_to_event,
                        "event_observed": event_observed,
                        "stage_end_horizon": spec["end_day"] - prediction_day,
                        "bin_exposure": exposure,
                        "event_bins": event_bins,
                        "stage_mask": stage_mask,
                    }
                )

        if self.max_samples and len(samples) > self.max_samples:
            indices = np.random.choice(len(samples), self.max_samples, replace=False)
            samples = [samples[index] for index in indices]
        if self.shuffle:
            np.random.shuffle(samples)

        events = sum(sample["event_observed"] for sample in samples)
        _rank0_print(
            f"[{self.split}] Survival samples={len(samples)}, events={events}, "
            f"censored={len(samples) - events}"
        )
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        df_followup = self._load_followup_data(sample)
        fname_key = sample["fname_key"]
        static_features = self._get_static_features(fname_key)
        patient_info = self.patient_info_map[fname_key]
        recipient_gender = patient_info["recipient_gender"]
        gender = "M" if str(recipient_gender).upper() in {"M", "MALE", "男"} else "F"
        dob = pd.to_datetime(patient_info["date_of_birth"], errors="coerce")
        if df_followup.empty:
            raise ValueError(f"No valid follow-up context for {fname_key}")

        first_row = df_followup.iloc[0]
        surgery_date = pd.to_datetime(first_row["报告日期"]) - pd.Timedelta(
            days=float(first_row["术后天数"])
        )
        age_years = (pd.to_datetime(first_row["报告日期"]) - dob).days / 365.25
        if not np.isfinite(age_years):
            age_years = 0.0
        age_years = max(0.0, age_years)
        task_info = deepcopy(self.task_schema["tacrolimus_abnormal_survival"])
        instruction, stage_window = build_survival_instruction(
            task_info,
            sample,
        )
        task_info.update(
            {
                "task": "tacrolimus_abnormal_survival",
                "stage_id": sample["stage_id"],
                "stage_window": stage_window,
                "prediction_day": sample["prediction_day"],
                "observed_day": sample["observed_day"],
                "time_to_event": sample["time_to_event"],
                "event_observed": sample["event_observed"],
                "stage_end_horizon": sample["stage_end_horizon"],
            }
        )

        final_table = self.structed_EHR_input_process(
            static_features=static_features,
            df_followup=df_followup,
            surgery_date=surgery_date,
            age_years=age_years,
            gender=gender,
        )
        survival_metadata = np.zeros(self.MAX_SURVIVAL_BINS, dtype=np.float32)
        survival_metadata[0] = sample["stage_end_horizon"]
        labels = torch.tensor(
            np.stack(
                [
                    sample["bin_exposure"],
                    sample["event_bins"],
                    sample["stage_mask"],
                    survival_metadata,
                ]
            ),
            dtype=torch.float32,
        )
        return {
            "idx": idx,
            "instruction": instruction,
            "input": "",
            "output": labels,
            "stage_id": sample["stage_id"],
            "task_info": task_info,
            "measurement_table": final_table,
        }


__all__ = [
    "EVENT_LABEL_COLUMN",
    "MAX_SURVIVAL_BINS",
    "STAGE_SPECS",
    "RenjiTacrolimusSurvivalDataset",
    "build_piecewise_survival_target",
    "build_survival_instruction",
    "load_patient_subset",
]
