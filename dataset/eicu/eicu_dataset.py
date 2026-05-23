import os
import sys
import json
import random
import time
import warnings
from collections import defaultdict
import multiprocessing as mp

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.eicu.task_info import get_task_info

warnings.filterwarnings("ignore")


_TABLE_LENGTH_WORKER_DATASET = None


def _eicu_table_length_cache_key_from_sample(sample_info):
    return (
        f"{sample_info.get('icustay_id', '')}|"
        f"{sample_info.get('task_name', '')}|"
        f"{sample_info.get('obs_hours', '')}"
    )


def _init_eicu_table_length_worker(processed_dir, obs_size):
    global _TABLE_LENGTH_WORKER_DATASET
    worker_dataset = EICUDataset.__new__(EICUDataset)
    worker_dataset.processed_dir = processed_dir
    worker_dataset.patient_dir = os.path.join(processed_dir, "patients")
    worker_dataset.obs_size = int(obs_size)
    _TABLE_LENGTH_WORKER_DATASET = worker_dataset


def _compute_eicu_table_length_worker(payload):
    idx, sample_info = payload
    dataset = _TABLE_LENGTH_WORKER_DATASET
    table_length = int(len(dataset.structed_EHR_input_process(sample_info)))
    cache_key = _eicu_table_length_cache_key_from_sample(sample_info)
    return idx, cache_key, table_length


class EICUDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        processed_dir=None,
        sample_info_path=None,
        sample_info=None,
        task_name=None,
        lazy_mode=False,
        shuffle=True,
        table_mode="text_only",
        max_samples=None,
        obs_size=12,
        gap_size=12,
        pred_size=24,
        return_meds=False,
        use_table_length_cache=False,
    ):
        random.seed(42)
        self.root_dir = root_dir
        self.processed_dir = processed_dir
        self.patient_dir = os.path.join(self.processed_dir, "patients")
        self.sample_info_path = sample_info_path
        self.lazy_mode = lazy_mode
        if table_mode not in {"text_only", "table_only", "table_plus_rest_text"}:
            raise ValueError(f"Unsupported table_mode: {table_mode}")
        self.table_mode = table_mode
        self.obs_size = int(obs_size)
        self.gap_size = int(gap_size)
        self.pred_size = int(pred_size)
        self.task_name = task_name
        self.task_schema = get_task_info()
        self.return_meds = return_meds
        self.use_table_length_cache = use_table_length_cache
        self.local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        self.table_length_cache_dir = os.path.join(self.processed_dir, "cache", "table_length")
        os.makedirs(self.table_length_cache_dir, exist_ok=True)
        sample_info_name = os.path.basename(self.sample_info_path) if self.sample_info_path else "sample_info_memory.json"
        self.table_length_cache_file = os.path.join(self.table_length_cache_dir, f"{sample_info_name}.table_length.json")

        if sample_info is None:
            with open(sample_info_path, "r", encoding="utf-8") as f:
                self.sample_info = json.load(f)
        else:
            self.sample_info = list(sample_info)

        if task_name is not None and task_name != "all":
            self.sample_info = [
                s
                for s in self.sample_info
                if s["task_name"] == task_name
            ]

        if self.use_table_length_cache and self.sample_info:
            self._ensure_table_lengths_cached()

        if shuffle:
            random.shuffle(self.sample_info)

        if max_samples is not None and max_samples < len(self.sample_info):
            self.sample_info = self._balance_samples(self.sample_info, max_samples)

        self.data = []
        if not self.lazy_mode:
            for idx in tqdm(range(len(self.sample_info)), desc="Pre-processing"):
                self.data.append(self._process_item(idx))

    def _table_length_cache_key(self, sample_info):
        return _eicu_table_length_cache_key_from_sample(sample_info)

    def _load_table_length_cache(self):
        if not os.path.exists(self.table_length_cache_file):
            return {}
        with open(self.table_length_cache_file, "r", encoding="utf-8") as f:
            return {str(k): int(v) for k, v in json.load(f).items()}

    def _save_table_length_cache(self, cache_data):
        with open(self.table_length_cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)

    def _ensure_table_lengths_cached(self):
        cache_data = self._load_table_length_cache()
        missing_indices = []

        for idx, sample_info in enumerate(self.sample_info):
            cache_key = self._table_length_cache_key(sample_info)
            if cache_key in cache_data:
                sample_info["table_length"] = int(cache_data[cache_key])
            else:
                missing_indices.append(idx)

        if not missing_indices:
            return

        if self.local_rank not in (-1, 0):
            wait_seconds = int(os.environ.get("EICU_TABLE_LENGTH_WAIT_SECONDS", "7200"))
            deadline = time.time() + max(1, wait_seconds)
            while time.time() < deadline:
                cache_data = self._load_table_length_cache()
                unresolved = 0
                for idx in missing_indices:
                    cache_key = self._table_length_cache_key(self.sample_info[idx])
                    if cache_key in cache_data:
                        self.sample_info[idx]["table_length"] = int(cache_data[cache_key])
                    else:
                        unresolved += 1
                if unresolved == 0:
                    return
                time.sleep(2)
            return

        tasks = [(idx, self.sample_info[idx]) for idx in missing_indices]
        requested_workers = int(os.environ.get("EICU_TABLE_LENGTH_WORKERS", str(os.cpu_count() or 1)))
        num_workers = max(1, min(requested_workers, len(tasks)))

        if num_workers == 1:
            _init_eicu_table_length_worker(self.processed_dir, self.obs_size)
            iterator = (_compute_eicu_table_length_worker(task) for task in tasks)
            for idx, cache_key, table_length in tqdm(iterator, total=len(tasks), desc="Computing eICU table_length"):
                self.sample_info[idx]["table_length"] = table_length
                cache_data[cache_key] = table_length
        else:
            chunk_size = max(1, min(int(os.environ.get("EICU_TABLE_LENGTH_CHUNK_SIZE", "64")), len(tasks)))
            with mp.get_context("fork").Pool(
                processes=num_workers,
                initializer=_init_eicu_table_length_worker,
                initargs=(self.processed_dir, self.obs_size),
            ) as pool:
                for idx, cache_key, table_length in tqdm(
                    pool.imap_unordered(_compute_eicu_table_length_worker, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc=f"Computing eICU table_length ({num_workers} workers)",
                ):
                    self.sample_info[idx]["table_length"] = table_length
                    cache_data[cache_key] = table_length

        self._save_table_length_cache(cache_data)

    def _balance_samples(self, all_samples, max_samples):
        if max_samples is None or max_samples >= len(all_samples):
            return all_samples

        label_groups = defaultdict(list)
        for sample in all_samples:
            lbl = str(sample["label"])
            label_groups[lbl].append(sample)

        if len(label_groups) == 0:
            return all_samples

        sorted_labels = sorted(label_groups.keys(), key=lambda k: len(label_groups[k]))
        balanced_samples = []
        remaining_quota = max_samples
        remaining_classes = len(sorted_labels)

        for lbl in sorted_labels:
            group = label_groups[lbl]
            random.shuffle(group)
            fair_share = remaining_quota // max(remaining_classes, 1)
            take_count = min(len(group), fair_share)
            balanced_samples.extend(group[:take_count])
            remaining_quota -= take_count
            remaining_classes -= 1

        random.shuffle(balanced_samples)
        return balanced_samples

    def _load_patient_data(self, icustay_id, obs_hours):
        patient_folder = os.path.join(self.patient_dir, str(icustay_id))
        if not os.path.isdir(patient_folder):
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        patient_file = os.path.join(patient_folder, "patient.csv")
        if not os.path.exists(patient_file):
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        patient_df = pd.read_csv(patient_file, low_memory=False)
        if patient_df.empty:
            return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        patient_info = patient_df.iloc[0]
        max_offset = int(obs_hours) * 60

        lab_file = os.path.join(patient_folder, "lab.csv")
        if os.path.exists(lab_file):
            lab_df = pd.read_csv(lab_file, low_memory=False)
            lab_df = lab_df[(lab_df["labresultoffset"] >= 0) & (lab_df["labresultoffset"] <= max_offset)].copy()
        else:
            lab_df = pd.DataFrame()

        med_file = os.path.join(patient_folder, "medication.csv")
        if os.path.exists(med_file):
            med_df = pd.read_csv(med_file, low_memory=False)
            med_df = med_df[(med_df["drugstartoffset"] >= 0) & (med_df["drugstartoffset"] <= max_offset)].copy()
        else:
            med_df = pd.DataFrame()

        infusion_file = os.path.join(patient_folder, "infusionDrug.csv")
        if os.path.exists(infusion_file):
            infusion_df = pd.read_csv(infusion_file, low_memory=False)
            infusion_df = infusion_df[
                (infusion_df["infusionoffset"] >= 0) & (infusion_df["infusionoffset"] <= max_offset)
            ].copy()
        else:
            infusion_df = pd.DataFrame()

        return patient_info, lab_df, med_df, infusion_df

    def _offset_to_datetime(self, offset_minutes):
        base_time = pd.Timestamp("2000-01-01 00:00:00")
        return (base_time + pd.Timedelta(minutes=float(offset_minutes))).strftime("%Y-%m-%d %H:%M:%S")


    def free_text_input_process(
        self,
        sample,
        include_table_sections=True,
        include_table_demographics=True,
        include_extra_demographics=True,
    ):
        icustay_id = sample.get("icustay_id")
        obs_hours = int(sample.get("obs_hours", self.obs_size))
        patient_info, lab_data, med_data, infusion_data = self._load_patient_data(icustay_id, obs_hours)

        sections = []
        if patient_info is not None:
            demo_lines = [
                "## Patient Demographics",
            ]
            if include_table_demographics:
                age = patient_info.get("age", "unknown")
                if age == "> 89":
                    age = ">89"
                demo_lines.extend([
                    f"- Age: {age} years",
                    f"- Gender: {patient_info.get('gender', 'unknown')}",
                ])
                ethnicity = patient_info.get("ethnicity", "unknown")
                if ethnicity and ethnicity != "unknown":
                    demo_lines.append(f"- Ethnicity: {ethnicity}")
            if include_extra_demographics:
                demo_lines.append(f"- Unit Type: {patient_info.get('unittype', 'unknown')}")
                admit_src = patient_info.get("unitadmitsource", "unknown")
                if admit_src and admit_src != "unknown":
                    demo_lines.append(f"- Admission Source: {admit_src}")
            if len(demo_lines) > 1:
                sections.append("\n".join(demo_lines))

        if include_table_sections and not lab_data.empty:
            lines = [
                "## Laboratory Tests",
                "| Time | Test Name | Value | Unit |",
                "| --- | --- | --- | --- |",
            ]
            for _, row in lab_data.sort_values("labresultoffset").iterrows():
                lines.append(
                    f"| {self._offset_to_datetime(row.get('labresultoffset', 0))} | "
                    f"{row.get('labname', 'Unknown')} | "
                    f"{row.get('labresult', '')} | "
                    f"{row.get('labmeasurenameinterface', '')} |"
                )
            sections.append("\n".join(lines))

        if include_table_sections and not med_data.empty:
            lines = [
                "## Medications",
                "| Time | Drug Name | Dosage | Frequency |",
                "| --- | --- | --- | --- |",
            ]
            for _, row in med_data.sort_values("drugstartoffset").iterrows():
                lines.append(
                    f"| {self._offset_to_datetime(row.get('drugstartoffset', 0))} | "
                    f"{row.get('drugname', 'unknown')} | "
                    f"{row.get('dosage', '')} | "
                    f"{row.get('frequency', '')} |"
                )
            sections.append("\n".join(lines))

        if include_table_sections and not infusion_data.empty:
            lines = [
                "## Infusion Drugs",
                "| Time | Drug Name | Rate | Amount |",
                "| --- | --- | --- | --- |",
            ]
            for _, row in infusion_data.sort_values("infusionoffset").iterrows():
                lines.append(
                    f"| {self._offset_to_datetime(row.get('infusionoffset', 0))} | "
                    f"{row.get('drugname', 'unknown')} | "
                    f"{row.get('infusionrate', '')} | "
                    f"{row.get('drugamount', '')} |"
                )
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def structed_EHR_input_process(self, sample):
        icustay_id = sample.get("icustay_id")
        obs_hours = int(sample.get("obs_hours", self.obs_size))
        patient_info, lab_data, med_data, infusion_data = self._load_patient_data(icustay_id, obs_hours)

        rows = []
        t0 = "2000-01-01 00:00:00"
        if patient_info is not None:
            age = str(patient_info.get("age", "unknown"))
            if age == "> 89":
                age = ">89"
            rows.append({"Time": t0, "Item": "Age", "Value": age, "Unit": "years", "Category": "person"})
            rows.append(
                {
                    "Time": t0,
                    "Item": "Gender",
                    "Value": str(patient_info.get("gender", "unknown")),
                    "Unit": "",
                    "Category": "person",
                }
            )
            ethnicity = str(patient_info.get("ethnicity", "unknown"))
            if ethnicity and ethnicity != "unknown":
                rows.append({"Time": t0, "Item": "Ethnicity", "Value": ethnicity, "Unit": "", "Category": "person"})

        for _, row in lab_data.iterrows():
            rows.append(
                {
                    "Time": self._offset_to_datetime(row.get("labresultoffset", 0)),
                    "Item": str(row.get("labname", "unknown")),
                    "Value": str(row.get("labresult", "")),
                    "Unit": str(row.get("labmeasurenameinterface", "")),
                    "Category": "measurement",
                }
            )

        for _, row in med_data.iterrows():
            rows.append(
                {
                    "Time": self._offset_to_datetime(row.get("drugstartoffset", 0)),
                    "Item": str(row.get("drugname", "unknown")),
                    "Value": str(row.get("dosage", "")),
                    "Unit": "",
                    "Category": "drug_exposure",
                }
            )

        for _, row in infusion_data.iterrows():
            rows.append(
                {
                    "Time": self._offset_to_datetime(row.get("infusionoffset", 0)),
                    "Item": str(row.get("drugname", "unknown")),
                    "Value": str(row.get("infusionrate", "")),
                    "Unit": "mL/hr",
                    "Category": "drug_exposure",
                }
            )

        measurement_df = pd.DataFrame(rows, columns=["Time", "Item", "Value", "Unit", "Category"])
        if not measurement_df.empty:
            measurement_df["Time"] = pd.to_datetime(measurement_df["Time"], errors="coerce")
            measurement_df = measurement_df.sort_values(by=["Time"]).reset_index(drop=True)
            measurement_df["Time"] = measurement_df["Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return measurement_df

    def _normalize_meds_fragment(self, value):
        text = "" if value is None else str(value).strip().upper()
        normalized = []
        for ch in text:
            if ch.isalnum():
                normalized.append(ch)
            else:
                normalized.append("_")
        text = "".join(normalized).strip("_")
        return text or "UNKNOWN"

    def meds_input_process(self, sample, return_hf_ehr_events=True):
        icustay_id = sample.get("icustay_id")
        obs_hours = int(sample.get("obs_hours", self.obs_size))
        patient_info, lab_data, med_data, infusion_data = self._load_patient_data(icustay_id, obs_hours)

        subject_id = (
            sample.get("patienthealthsystemstayid")
            or sample.get("subject_id")
            or sample.get("icustay_id")
            or icustay_id
        )
        subject_id = "" if subject_id is None else str(subject_id)

        rows = []
        t0 = pd.Timestamp("2000-01-01 00:00:00")
        if patient_info is not None:
            age = patient_info.get("age", "")
            age_clean = "" if pd.isna(age) else str(age).strip()
            if age_clean and age_clean.lower() != "unknown":
                numeric_age = pd.to_numeric(age_clean.replace("> ", ">").replace(">", ""), errors="coerce")
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": t0,
                        "code": "AGE",
                        "numeric_value": float(numeric_age) if pd.notna(numeric_age) else None,
                        "text_value": "" if pd.notna(numeric_age) else age_clean,
                        "unit": "years",
                        "omop_table": "person",
                    }
                )

            gender = patient_info.get("gender", "")
            if not pd.isna(gender):
                gender_norm = self._normalize_meds_fragment(gender)
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": t0,
                        "code": f"GENDER//{gender_norm}",
                        "numeric_value": None,
                        "text_value": "",
                        "unit": "",
                        "omop_table": "person",
                    }
                )

            ethnicity = patient_info.get("ethnicity", "")
            ethnicity_clean = "" if pd.isna(ethnicity) else str(ethnicity).strip()
            if ethnicity_clean and ethnicity_clean.lower() != "unknown":
                rows.append(
                    {
                        "subject_id": subject_id,
                        "time": t0,
                        "code": f"RACE//{self._normalize_meds_fragment(ethnicity_clean)}",
                        "numeric_value": None,
                        "text_value": "",
                        "unit": "",
                        "omop_table": "person",
                    }
                )

        for _, row in lab_data.iterrows():
            lab_name = row.get("labname", "unknown")
            lab_system = row.get("labmeasurenamesystem", "")
            code_parts = ["LAB", self._normalize_meds_fragment(lab_name)]
            if not pd.isna(lab_system) and str(lab_system).strip():
                code_parts.append(self._normalize_meds_fragment(lab_system))
            code = "//".join(code_parts)

            lab_time = pd.to_datetime(self._offset_to_datetime(row.get("labresultoffset", 0)), errors="coerce")
            lab_result = row.get("labresult", None)
            lab_text = row.get("labresulttext", "")
            lab_unit = row.get("labmeasurenameinterface", "")

            numeric_value = pd.to_numeric(lab_result, errors="coerce")
            text_value = ""
            if pd.isna(numeric_value):
                if not pd.isna(lab_result):
                    text_value = str(lab_result).strip()
                if (not text_value) and (not pd.isna(lab_text)):
                    text_value = str(lab_text).strip()

            rows.append(
                {
                    "subject_id": subject_id,
                    "time": lab_time,
                    "code": code,
                    "numeric_value": float(numeric_value) if pd.notna(numeric_value) else None,
                    "text_value": text_value,
                    "unit": "" if pd.isna(lab_unit) else str(lab_unit).strip(),
                    "omop_table": "measurement",
                }
            )

        for _, row in med_data.iterrows():
            med_time = pd.to_datetime(self._offset_to_datetime(row.get("drugstartoffset", 0)), errors="coerce")
            drug_name = row.get("drugname", "unknown")
            code = f"MEDICATION//STARTED//{self._normalize_meds_fragment(drug_name)}"

            dosage = row.get("dosage", None)
            numeric_value = pd.to_numeric(dosage, errors="coerce")
            text_value = ""
            if pd.isna(numeric_value) and (not pd.isna(dosage)):
                text_value = str(dosage).strip()

            rows.append(
                {
                    "subject_id": subject_id,
                    "time": med_time,
                    "code": code,
                    "numeric_value": float(numeric_value) if pd.notna(numeric_value) else None,
                    "text_value": text_value,
                    "unit": "",
                    "omop_table": "drug_exposure",
                }
            )

        for _, row in infusion_data.iterrows():
            infusion_time = pd.to_datetime(
                self._offset_to_datetime(row.get("infusionoffset", 0)),
                errors="coerce",
            )
            drug_name = row.get("drugname", "unknown")
            code = f"INFUSION_DRUG//{self._normalize_meds_fragment(drug_name)}"

            infusion_rate = row.get("infusionrate", None)
            numeric_value = pd.to_numeric(infusion_rate, errors="coerce")
            text_value = ""
            if pd.isna(numeric_value):
                if not pd.isna(infusion_rate):
                    text_value = str(infusion_rate).strip()
            if not text_value:
                drug_amount = row.get("drugamount", None)
                if not pd.isna(drug_amount):
                    text_value = str(drug_amount).strip()

            rows.append(
                {
                    "subject_id": subject_id,
                    "time": infusion_time,
                    "code": code,
                    "numeric_value": float(numeric_value) if pd.notna(numeric_value) else None,
                    "text_value": text_value,
                    "unit": "mL/hr",
                    "omop_table": "drug_exposure",
                }
            )

        meds_df = pd.DataFrame(
            rows,
            columns=["subject_id", "time", "code", "numeric_value", "text_value", "unit", "omop_table"],
        )
        if not meds_df.empty:
            meds_df["time"] = pd.to_datetime(meds_df["time"], errors="coerce")
            meds_df = meds_df.sort_values(by=["time"]).reset_index(drop=True)
            meds_df["time"] = meds_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

        meds_events = []
        for row in meds_df.to_dict(orient="records"):
            code = str(row.get("code", "")).strip()
            if not code:
                continue

            event = {"code": code}

            t = str(row.get("time", "")).strip()
            if t:
                event["start"] = t
                event["end"] = t

            num_val = row.get("numeric_value")
            txt_val = str(row.get("text_value", "")).strip()
            if pd.notna(num_val):
                event["value"] = float(num_val)
            elif txt_val:
                event["value"] = txt_val

            unit = str(row.get("unit", "")).strip()
            if unit:
                event["unit"] = unit

            table_name = str(row.get("omop_table", "")).strip()
            if table_name:
                event["omop_table"] = table_name

            meds_events.append(event)

        if not return_hf_ehr_events:
            return meds_df, meds_events

        try:
            from hf_ehr.config import Event
        except Exception:
            return meds_df, meds_events, None

        hf_ehr_events = []
        for event in meds_events:
            kwargs = {"code": event["code"]}
            for key in ("value", "unit", "start", "end", "omop_table"):
                if key in event and event[key] not in (None, ""):
                    kwargs[key] = event[key]
            hf_ehr_events.append(Event(**kwargs))

        return meds_df, meds_events, hf_ehr_events

    def MEDS_input_process(self, sample, return_hf_ehr_events=True):
        return self.meds_input_process(
            sample,
            return_hf_ehr_events=return_hf_ehr_events,
        )

    def _process_item(self, index):
        sample = self.sample_info[index]
        if sample.get("task") == "pretraining_context":
            context = "" if self.table_mode == "table_only" else self.free_text_input_process(sample)
            output_sample = {
                "idx": index,
                "input": context,
                "output": "",
                "task_info": {"task": "pretraining_context"},
                "instruction": "",
            }
            if self.table_mode in {"table_only", "table_plus_rest_text"}:
                measurement_df = self.structed_EHR_input_process(sample)
                output_sample["measurement_table"] = measurement_df
                output_sample["table_length"] = len(measurement_df)
                if self.table_mode == "table_plus_rest_text":
                    output_sample["remaining_text"] = self.free_text_input_process(
                        sample,
                        include_table_sections=False,
                        include_table_demographics=False,
                        include_extra_demographics=True,
                    )
            return output_sample

        label = sample["label"]

        context = "" if self.table_mode == "table_only" else self.free_text_input_process(sample)
        task_info = self.task_schema[self.task_name]
        output_sample = {
            "idx": index,
            "input": context,
            "output": label if isinstance(label, str) else str(label),
            "task_info": task_info,
            "instruction": task_info["instruction"],
        }

        if self.return_meds:
            meds_df, meds_events, hf_ehr_events = self.meds_input_process(
                sample,
                return_hf_ehr_events=True,
            )
            output_sample["meds_table"] = meds_df
            output_sample["meds_events"] = meds_events
            if hf_ehr_events is not None:
                output_sample["hf_ehr_events"] = hf_ehr_events

        if self.table_mode in {"table_only", "table_plus_rest_text"}:
            measurement_df = self.structed_EHR_input_process(sample)
            output_sample["measurement_table"] = measurement_df
            output_sample["table_length"] = len(measurement_df)
            if self.table_mode == "table_plus_rest_text":
                output_sample["remaining_text"] = self.free_text_input_process(
                    sample,
                    include_table_sections=False,
                    include_table_demographics=False,
                    include_extra_demographics=True,
                )
        return output_sample

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, index):
        if self.lazy_mode:
            sample = self._process_item(index)
        else:
            sample = self.data[index]

        if self.return_meds and "meds_events" not in sample:
            meds_df, meds_events, hf_ehr_events = self.meds_input_process(
                self.sample_info[index],
                return_hf_ehr_events=True,
            )
            sample["meds_table"] = meds_df
            sample["meds_events"] = meds_events
            if hf_ehr_events is not None:
                sample["hf_ehr_events"] = hf_ehr_events

        return sample


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="eICU dataset template usage example.")
    parser.add_argument("--root_dir", type=str, default="/home/ma-user/sfs_turbo/Data/eicu-crd/2.0")
    parser.add_argument("--processed_dir", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed")
    parser.add_argument("--sample_info_path", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed/sample_info_test.json")
    parser.add_argument("--task_name", type=str, default="mortality")
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--lazy_mode", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "eicu",
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset_text = EICUDataset(
        root_dir=args.root_dir,
        processed_dir=args.processed_dir,
        sample_info_path=args.sample_info_path,
        task_name=args.task_name,
        lazy_mode=args.lazy_mode,
        shuffle=False,
        table_mode="text_only",
        max_samples=args.max_samples,
    )
    dataset_struct = EICUDataset(
        root_dir=args.root_dir,
        processed_dir=args.processed_dir,
        sample_info_path=args.sample_info_path,
        task_name=args.task_name,
        lazy_mode=args.lazy_mode,
        shuffle=False,
        table_mode="table_only",
        max_samples=args.max_samples,
    )
    dataset_mixed = EICUDataset(
        root_dir=args.root_dir,
        processed_dir=args.processed_dir,
        sample_info_path=args.sample_info_path,
        task_name=args.task_name,
        lazy_mode=args.lazy_mode,
        shuffle=False,
        table_mode="table_plus_rest_text",
        max_samples=args.max_samples,
    )

    print(f"dataset_text size: {len(dataset_text)}")
    print(f"dataset_struct size: {len(dataset_struct)}")
    print(f"dataset_mixed size: {len(dataset_mixed)}")
    if len(dataset_text) == 0:
        raise SystemExit("No samples found.")

    idx = max(0, min(args.sample_index, len(dataset_text) - 1))
    sample_text = dataset_text[idx]
    sample_struct = dataset_struct[idx]
    sample_mixed = dataset_mixed[idx]

    print(f"\nSample index: {idx}")
    print(f"text keys: {list(sample_text.keys())}")
    print(f"struct keys: {list(sample_struct.keys())}")
    print(f"mixed keys: {list(sample_mixed.keys())}")

    text_out = os.path.join(args.out_dir, "eicu_text_only_sample.txt")
    with open(text_out, "w", encoding="utf-8") as f:
        f.write(str(sample_text.get("input", "")) + "\n")
    print(f"Saved text sample: {text_out}")

    struct_text_out = os.path.join(args.out_dir, "eicu_table_only_text_sample.txt")
    with open(struct_text_out, "w", encoding="utf-8") as f:
        f.write(str(sample_struct.get("input", "")) + "\n")
    print(f"Saved structured-text sample: {struct_text_out}")

    mixed_text_out = os.path.join(args.out_dir, "eicu_table_plus_rest_text_sample.txt")
    with open(mixed_text_out, "w", encoding="utf-8") as f:
        f.write(str(sample_mixed.get("remaining_text", "")) + "\n")
    print(f"Saved mixed-text sample: {mixed_text_out}")

    table = sample_struct.get("measurement_table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        table_out = os.path.join(args.out_dir, "eicu_table_only_sample.csv")
        table.to_csv(table_out, index=False, encoding="utf-8-sig")
        print(f"Saved structured table sample: {table_out} (shape={table.shape})")
    else:
        print("No measurement_table found in structured sample.")
