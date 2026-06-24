import argparse
import concurrent.futures as futures
import csv
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic.mimic_dataset import read_parquet
from dataset.mimic.input_format import safe_read

_MIMIC_WORKER_EHR_DIR = None
_EHRSHOT_WORKER_ROOT_DIR = None


MIMIC_TTE_TASKS = {
    "ED_Inpatient_Mortality": ("Time_to_Inpatient_Mortality_after_ED", 30.0),
    "ED_ICU_Tranfer_12hour": ("Time_to_ICU_Transfer_after_ED", 0.5),
    "ED_Reattendance_3day": ("Time_to_ED_Reattendance", 3.0),
    "ED_Critical_Outcomes": ("Time_to_ED_Critical_Outcome", 30.0),
    "Readmission_30day": ("Time_to_Hospital_Readmission", 30.0),
    "Readmission_60day": ("Time_to_Hospital_Readmission", 60.0),
    "Inpatient_Mortality": ("Time_to_Inpatient_Mortality", 30.0),
    "LengthOfStay_3day": ("Time_to_Hospital_Discharge", 30.0),
    "LengthOfStay_7day": ("Time_to_Hospital_Discharge", 30.0),
    "ICU_Mortality_1day": ("Time_to_ICU_Mortality", 1.0),
    "ICU_Mortality_2day": ("Time_to_ICU_Mortality", 2.0),
    "ICU_Mortality_3day": ("Time_to_ICU_Mortality", 3.0),
    "ICU_Mortality_7day": ("Time_to_ICU_Mortality", 7.0),
    "ICU_Mortality_14day": ("Time_to_ICU_Mortality", 14.0),
    "ICU_Stay_7day": ("Time_to_ICU_Discharge", 30.0),
    "ICU_Stay_14day": ("Time_to_ICU_Discharge", 30.0),
    "ICU_Readmission": ("Time_to_ICU_Readmission", 30.0),
}
MIMIC_DEATH_SOURCE_TASKS = {
    "ED_Inpatient_Mortality",
    "Inpatient_Mortality",
    "ICU_Mortality_1day",
    "ICU_Mortality_2day",
    "ICU_Mortality_3day",
    "ICU_Mortality_7day",
    "ICU_Mortality_14day",
}

EICU_TTE_TASKS = {
    "mortality": ("Time_to_ICU_or_Hospital_Mortality", 1.0),
    "long_term_mortality": ("Time_to_Long_Term_Mortality", 14.0),
    "los_3day": ("Time_to_ICU_Discharge", 30.0),
    "los_7day": ("Time_to_ICU_Discharge", 30.0),
}
EICU_DEATH_SOURCE_TASKS = {"mortality", "long_term_mortality"}

EHRSHOT_TTE_TASKS = {
    "guo_los": ("Time_to_Hospital_Discharge", 30.0),
    "guo_readmission": ("Time_to_Hospital_Readmission", 30.0),
    "guo_icu": ("Time_to_ICU_Transfer", 30.0),
}


def parse_time(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def fmt_time(value: Optional[datetime]) -> str:
    return "" if value is None else value.strftime("%Y-%m-%d %H:%M:%S")


def event_start(event: Dict[str, Any]) -> Optional[datetime]:
    return parse_time(event.get("starttime"))


def item_time(event: Dict[str, Any], key: str) -> Optional[datetime]:
    items = event.get("items") or []
    if not items:
        return None
    return parse_time(items[0].get(key))


def time_to_days(start: datetime, end: datetime) -> float:
    return max((end - start).total_seconds() / 86400.0, 0.0)


def add_duration_fields(
    row: Dict[str, Any],
    prediction_time: datetime,
    event_time: Optional[datetime],
    censor_time: Optional[datetime],
    event_observed: bool,
    horizon_days: float,
) -> Optional[Dict[str, Any]]:
    horizon_time = prediction_time + timedelta(days=float(horizon_days))
    if event_time is not None and event_time <= prediction_time:
        return None
    if event_observed and event_time is not None and event_time <= horizon_time:
        observed_time = event_time
        observed = 1
    else:
        observed_time = min(
            [time for time in (censor_time, horizon_time) if time is not None]
        )
        observed = 0
    if observed_time <= prediction_time:
        return None
    row.update(
        {
            "prediction_time": fmt_time(prediction_time),
            "event_time": fmt_time(event_time if observed else None),
            "censor_time": fmt_time(observed_time if not observed else censor_time),
            "time_to_event": f"{time_to_days(prediction_time, observed_time):.6f}",
            "event_observed": observed,
            "horizon_days": f"{float(horizon_days):.6f}",
            "time_unit": "day",
        }
    )
    return row


def first_after(
    trajectory: List[Dict[str, Any]],
    start_idx: int,
    file_name: str,
    hadm_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    for event in trajectory[start_idx + 1 :]:
        if event.get("file_name") != file_name:
            continue
        if hadm_id is not None and str(safe_read(event.get("hadm_id"))) != str(hadm_id):
            continue
        return event
    return None


def same_hadm_last_time(
    trajectory: List[Dict[str, Any]], start_idx: int, hadm_id: Optional[str]
) -> Optional[datetime]:
    last_time = None
    for event in trajectory[start_idx + 1 :]:
        if str(safe_read(event.get("hadm_id"))) != str(hadm_id):
            continue
        current = event_start(event)
        if current is not None:
            last_time = current
    return last_time


def build_mimic_tte_row(
    source: Dict[str, str],
    trajectory: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    source_task = str(source["task"])
    if source_task not in MIMIC_TTE_TASKS:
        return None
    task_name, horizon_days = MIMIC_TTE_TASKS[source_task]
    context_begin = int(float(source["context_begin"]))
    context_end = int(float(source["context_end"]))
    anchor_idx = context_begin
    if source.get("event") == "discharge":
        anchor_idx = context_end - 1
    if anchor_idx < 0 or anchor_idx >= len(trajectory):
        return None

    anchor = trajectory[anchor_idx]
    hadm_id = str(safe_read(source.get("hadm_id") or anchor.get("hadm_id")))
    patient_dod = parse_time((trajectory[0].get("items") or [{}])[0].get("dod"))
    prediction_time = None
    event_time = None
    censor_time = None
    event_observed = False

    if source_task.startswith("ED_"):
        prediction_time = item_time(anchor, "outtime")
        icu_event = first_after(trajectory, anchor_idx, "icustays", hadm_id)
        next_ed = first_after(trajectory, anchor_idx, "edstays")
        death_time = patient_dod
        last_time = same_hadm_last_time(trajectory, anchor_idx, hadm_id)
        if source_task == "ED_Inpatient_Mortality":
            event_time = death_time
            event_observed = bool(death_time and (last_time is None or death_time <= last_time))
            censor_time = last_time
        elif source_task == "ED_ICU_Tranfer_12hour":
            event_time = event_start(icu_event) if icu_event else None
            event_observed = event_time is not None
            censor_time = prediction_time + timedelta(days=horizon_days) if prediction_time else None
        elif source_task == "ED_Reattendance_3day":
            event_time = event_start(next_ed) if next_ed else None
            event_observed = event_time is not None
            censor_time = prediction_time + timedelta(days=horizon_days) if prediction_time else None
        else:
            candidates = []
            icu_time = event_start(icu_event) if icu_event else None
            if icu_time is not None:
                candidates.append(icu_time)
            if death_time is not None and (last_time is None or death_time <= last_time):
                candidates.append(death_time)
            event_time = min(candidates) if candidates else None
            event_observed = event_time is not None
            censor_time = last_time
    elif source_task.startswith("Readmission_"):
        prediction_time = event_start(anchor)
        next_admission = first_after(trajectory, anchor_idx, "admissions")
        event_time = event_start(next_admission) if next_admission else None
        event_observed = event_time is not None
        censor_time = prediction_time + timedelta(days=horizon_days) if prediction_time else None
    elif source_task in {"Inpatient_Mortality", "LengthOfStay_3day", "LengthOfStay_7day"}:
        prediction_event = trajectory[context_end - 1]
        prediction_time = event_start(prediction_event)
        discharge_time = item_time(anchor, "dischtime")
        if source_task == "Inpatient_Mortality":
            event_time = patient_dod
            event_observed = bool(event_time and discharge_time and event_time <= discharge_time)
            censor_time = discharge_time
        else:
            event_time = discharge_time
            event_observed = event_time is not None
            censor_time = discharge_time
    elif source_task.startswith("ICU_"):
        prediction_event = trajectory[context_end - 1]
        prediction_time = event_start(prediction_event)
        icu_out = item_time(anchor, "outtime")
        next_icu = first_after(trajectory, anchor_idx, "icustays", hadm_id)
        if source_task.startswith("ICU_Mortality_"):
            event_time = patient_dod
            event_observed = bool(event_time and icu_out and event_time <= icu_out)
            censor_time = icu_out
        elif source_task.startswith("ICU_Stay_"):
            event_time = icu_out
            event_observed = event_time is not None
            censor_time = icu_out
        else:
            event_time = event_start(next_icu) if next_icu else None
            event_observed = event_time is not None
            censor_time = icu_out

    if prediction_time is None:
        return None
    row = dict(source)
    row["source_binary_task"] = source_task
    row["task"] = task_name
    row["target"] = "tte"
    return add_duration_fields(
        row, prediction_time, event_time, censor_time, event_observed, horizon_days
    )


def init_mimic_worker(mimic_ehr_dir: str):
    global _MIMIC_WORKER_EHR_DIR
    _MIMIC_WORKER_EHR_DIR = mimic_ehr_dir


def process_mimic_record(payload):
    source_task, source = payload
    subject_id = str(source["subject_id"])
    trajectory = read_parquet(
        os.path.join(_MIMIC_WORKER_EHR_DIR, f"{subject_id}.parquet")
    )
    return build_mimic_tte_row(source, trajectory)


def process_mimic_subject_group(
    payload: Tuple[str, List[Tuple[str, Dict[str, str]]]]
) -> List[Optional[Dict[str, Any]]]:
    subject_id, records = payload
    trajectory = read_parquet(
        os.path.join(_MIMIC_WORKER_EHR_DIR, f"{subject_id}.parquet")
    )
    return [build_mimic_tte_row(source, trajectory) for _, source in records]


def read_csv_records(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_mimic_ehr_dir(path: str) -> str:
    if os.path.isdir(path) and glob_has_parquet(path):
        return path
    candidates = [
        os.path.join(path, "patients_ehr"),
        path.replace("/EHR", "/patients_ehr"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate) and glob_has_parquet(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find MIMIC per-patient parquet directory from {path}. "
        "Expected a directory such as .../mimic-iv-3.1_tabular/patients_ehr."
    )


def glob_has_parquet(path: str) -> bool:
    try:
        with os.scandir(path) as entries:
            return any(entry.name.endswith(".parquet") for entry in entries)
    except FileNotFoundError:
        return False


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_records_by_key(records: Iterable[Any], key_fn) -> List[Tuple[str, List[Any]]]:
    grouped: Dict[str, List[Any]] = {}
    for record in records:
        key = str(key_fn(record))
        grouped.setdefault(key, []).append(record)
    return list(grouped.items())


def build_mimic(args, split: str):
    index_dir = args.mimic_train_index_dir if split == "train" else args.mimic_val_index_dir
    if not index_dir or not os.path.isdir(index_dir):
        return
    mimic_ehr_dir = resolve_mimic_ehr_dir(args.mimic_ehr_dir)
    out_dir = os.path.join(args.output_dir, "mimic_iv", split)
    emitted = set()
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    records = []
    source_tasks = (
        [task for task in MIMIC_TTE_TASKS if task in MIMIC_DEATH_SOURCE_TASKS]
        if args.death_only
        else list(MIMIC_TTE_TASKS)
    )
    for source_task in source_tasks:
        path = os.path.join(index_dir, f"{source_task}.csv")
        if not os.path.exists(path):
            continue
        records.extend((source_task, source) for source in read_csv_records(path))

    grouped_records = group_records_by_key(
        records,
        lambda record: record[1]["subject_id"],
    )
    worker_count = min(max(1, int(args.num_workers)), max(1, len(grouped_records)))
    iterator = None
    progress = tqdm(
        total=len(records),
        desc=f"mimic_iv {split}",
        unit="sample",
        dynamic_ncols=True,
    )
    try:
        if worker_count <= 1:
            init_mimic_worker(mimic_ehr_dir)
            iterator = map(process_mimic_subject_group, grouped_records)
        else:
            executor = futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=init_mimic_worker,
                initargs=(mimic_ehr_dir,),
            )
            iterator = executor.map(
                process_mimic_subject_group,
                grouped_records,
                chunksize=max(1, int(args.worker_chunksize)),
            )
        for group_rows, (_, group_records) in zip(iterator, grouped_records):
            progress.update(len(group_records))
            for row in group_rows:
                if row is None:
                    continue
                key = (
                    row.get("subject_id"),
                    row.get("hadm_id"),
                    row.get("task"),
                    row.get("source_binary_task"),
                    row.get("prediction_time"),
                    row.get("horizon_days"),
                )
                if key in emitted:
                    continue
                emitted.add(key)
                rows_by_task.setdefault(row["task"], []).append(row)
    finally:
        progress.close()
        if worker_count > 1 and "executor" in locals():
            executor.shutdown()
    for task_name, rows in rows_by_task.items():
        write_csv(os.path.join(out_dir, f"{task_name}.csv"), rows)
        print(f"mimic_iv {split} {task_name}: {len(rows)}")


def synthetic_eicu_time(offset_minutes: float) -> datetime:
    return datetime(2000, 1, 1) + timedelta(minutes=float(offset_minutes))


def load_json_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_eicu(args, split: str):
    sample_path = args.eicu_train_sample_info_path if split == "train" else args.eicu_val_sample_info_path
    if not sample_path or not os.path.exists(sample_path) or not os.path.exists(args.eicu_cohorts_path):
        return
    cohorts = pd.read_csv(args.eicu_cohorts_path)
    cohorts = cohorts.set_index("patientunitstayid").to_dict(orient="index")
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    samples = [
        sample
        for sample in load_json_records(sample_path)
        if sample.get("task_name") in EICU_TTE_TASKS
        and (
            not args.death_only
            or sample.get("task_name") in EICU_DEATH_SOURCE_TASKS
        )
    ]
    for sample in tqdm(
        samples,
        desc=f"eicu {split}",
        unit="sample",
        dynamic_ncols=True,
    ):
        source_task = sample.get("task_name")
        stay_id = int(sample["icustay_id"])
        cohort = cohorts.get(stay_id)
        if cohort is None:
            continue
        task_name, horizon_days = EICU_TTE_TASKS[source_task]
        obs = float(sample.get("obs_hours", 12))
        gap = float(sample.get("gap_hours", 0))
        prediction_offset = (obs + gap) * 60.0
        prediction_time = synthetic_eicu_time(prediction_offset)
        out_offset = cohort.get("OUTTIME")
        disch_offset = cohort.get("DISCHTIME", out_offset)
        death = bool(cohort.get("IN_ICU_MORTALITY")) or str(
            cohort.get("HOS_DISCHARGE_LOCATION", "")
        ).lower() == "death"
        if source_task in {"mortality", "long_term_mortality"}:
            event_time = synthetic_eicu_time(disch_offset) if death else None
            censor_time = synthetic_eicu_time(disch_offset)
            event_observed = death
        else:
            event_time = synthetic_eicu_time(out_offset)
            censor_time = event_time
            event_observed = True
        row = {
            "icustay_id": sample["icustay_id"],
            "patient_id": sample.get("patient_id", ""),
            "task_name": task_name,
            "source_binary_task": source_task,
            "label": "tte",
            "split": split,
            "obs_hours": sample.get("obs_hours", ""),
            "gap_hours": sample.get("gap_hours", ""),
            "pred_hours": sample.get("pred_hours", ""),
        }
        row = add_duration_fields(
            row, prediction_time, event_time, censor_time, event_observed, horizon_days
        )
        if row is not None:
            rows_by_task.setdefault(task_name, []).append(row)
    out_dir = os.path.join(args.output_dir, "eicu", split)
    for task_name, rows in rows_by_task.items():
        write_csv(os.path.join(out_dir, f"{task_name}.csv"), rows)
        print(f"eicu {split} {task_name}: {len(rows)}")


def load_ehrshot_patient(root_dir: str, patient_id: str) -> List[Dict[str, str]]:
    return read_csv_records(os.path.join(root_dir, "patient_ehr", f"{patient_id}.csv"))


def init_ehrshot_worker(root_dir: str):
    global _EHRSHOT_WORKER_ROOT_DIR
    _EHRSHOT_WORKER_ROOT_DIR = root_dir


def ehrshot_tte_row(args, source: Dict[str, str]) -> Optional[Dict[str, Any]]:
    source_task = source.get("task_name")
    if source_task not in EHRSHOT_TTE_TASKS:
        return None
    task_name, horizon_days = EHRSHOT_TTE_TASKS[source_task]
    prediction_time = parse_time(source.get("prediction_time"))
    if prediction_time is None:
        return None
    patient_rows = load_ehrshot_patient(args.ehrshot_root_dir, str(source["patient_id"]))
    horizon_time = prediction_time + timedelta(days=horizon_days)
    event_time = None
    censor_time = horizon_time
    if source_task == "guo_los":
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            desc = f"{row.get('code', '')} {row.get('description', '')}".lower()
            if "inpatient" not in desc and "visit/ip" not in desc:
                continue
            start = parse_time(row.get("start"))
            end = parse_time(row.get("end"))
            if start and end and start <= prediction_time <= end:
                event_time = end
                censor_time = end
                break
    elif source_task == "guo_readmission":
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            desc = f"{row.get('code', '')} {row.get('description', '')}".lower()
            if "inpatient" not in desc and "visit/ip" not in desc:
                continue
            start = parse_time(row.get("start"))
            if start and prediction_time < start <= horizon_time:
                event_time = start
                censor_time = horizon_time
                break
    else:
        for row in patient_rows:
            text = f"{row.get('code', '')} {row.get('description', '')}".lower()
            start = parse_time(row.get("start"))
            if start and prediction_time < start <= horizon_time and "icu" in text:
                event_time = start
                censor_time = horizon_time
                break
    observed = event_time is not None and str(source.get("label", "")).lower() == "true"
    out = dict(source)
    out["task_name"] = task_name
    out["source_binary_task"] = source_task
    out["label"] = "tte"
    return add_duration_fields(out, prediction_time, event_time, censor_time, observed, horizon_days)


def ehrshot_tte_row_from_root(root_dir: str, source: Dict[str, str]) -> Optional[Dict[str, Any]]:
    source_task = source.get("task_name")
    if source_task not in EHRSHOT_TTE_TASKS:
        return None
    task_name, horizon_days = EHRSHOT_TTE_TASKS[source_task]
    prediction_time = parse_time(source.get("prediction_time"))
    if prediction_time is None:
        return None
    patient_rows = load_ehrshot_patient(root_dir, str(source["patient_id"]))
    horizon_time = prediction_time + timedelta(days=horizon_days)
    event_time = None
    censor_time = horizon_time
    if source_task == "guo_los":
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            desc = f"{row.get('code', '')} {row.get('description', '')}".lower()
            if "inpatient" not in desc and "visit/ip" not in desc:
                continue
            start = parse_time(row.get("start"))
            end = parse_time(row.get("end"))
            if start and end and start <= prediction_time <= end:
                event_time = end
                censor_time = end
                break
    elif source_task == "guo_readmission":
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            desc = f"{row.get('code', '')} {row.get('description', '')}".lower()
            if "inpatient" not in desc and "visit/ip" not in desc:
                continue
            start = parse_time(row.get("start"))
            if start and prediction_time < start <= horizon_time:
                event_time = start
                censor_time = horizon_time
                break
    else:
        for row in patient_rows:
            text = f"{row.get('code', '')} {row.get('description', '')}".lower()
            start = parse_time(row.get("start"))
            if start and prediction_time < start <= horizon_time and "icu" in text:
                event_time = start
                censor_time = horizon_time
                break
    observed = event_time is not None and str(source.get("label", "")).lower() == "true"
    out = dict(source)
    out["task_name"] = task_name
    out["source_binary_task"] = source_task
    out["label"] = "tte"
    return add_duration_fields(out, prediction_time, event_time, censor_time, observed, horizon_days)


def ehrshot_tte_row_from_patient_rows(
    source: Dict[str, str],
    patient_rows: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    source_task = source.get("task_name")
    if source_task not in EHRSHOT_TTE_TASKS:
        return None
    task_name, horizon_days = EHRSHOT_TTE_TASKS[source_task]
    prediction_time = parse_time(source.get("prediction_time"))
    if prediction_time is None:
        return None
    horizon_time = prediction_time + timedelta(days=horizon_days)
    event_time = None
    censor_time = horizon_time
    if source_task == "guo_los":
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            desc = f"{row.get('code', '')} {row.get('description', '')}".lower()
            if "inpatient" not in desc and "visit/ip" not in desc:
                continue
            start = parse_time(row.get("start"))
            end = parse_time(row.get("end"))
            if start and end and start <= prediction_time <= end:
                event_time = end
                censor_time = end
                break
    elif source_task == "guo_readmission":
        for row in patient_rows:
            if row.get("omop_table") != "visit_occurrence":
                continue
            desc = f"{row.get('code', '')} {row.get('description', '')}".lower()
            if "inpatient" not in desc and "visit/ip" not in desc:
                continue
            start = parse_time(row.get("start"))
            if start and prediction_time < start <= horizon_time:
                event_time = start
                censor_time = horizon_time
                break
    else:
        for row in patient_rows:
            text = f"{row.get('code', '')} {row.get('description', '')}".lower()
            start = parse_time(row.get("start"))
            if start and prediction_time < start <= horizon_time and "icu" in text:
                event_time = start
                censor_time = horizon_time
                break
    observed = event_time is not None and str(source.get("label", "")).lower() == "true"
    out = dict(source)
    out["task_name"] = task_name
    out["source_binary_task"] = source_task
    out["label"] = "tte"
    return add_duration_fields(out, prediction_time, event_time, censor_time, observed, horizon_days)


def ehrshot_patient_event_times(rows: List[Dict[str, str]]):
    death_time = None
    last_time = None
    for row in rows:
        for key in ("start", "end"):
            current_time = parse_time(row.get(key))
            if current_time is not None and (last_time is None or current_time > last_time):
                last_time = current_time
        table = str(row.get("omop_table", "")).strip().lower()
        description = str(row.get("description", "")).strip().lower()
        if table == "death" or "patient status \"deceased\"" in description:
            current_time = parse_time(row.get("start")) or parse_time(row.get("end"))
            if current_time is not None and (death_time is None or current_time < death_time):
                death_time = current_time
    return death_time, last_time


def ehrshot_death_row_from_root(
    root_dir: str,
    source: Dict[str, str],
    horizon_days: float,
) -> Optional[Dict[str, Any]]:
    prediction_time = parse_time(source.get("prediction_time"))
    if prediction_time is None:
        return None
    patient_rows = load_ehrshot_patient(root_dir, str(source["patient_id"]))
    death_time, last_time = ehrshot_patient_event_times(patient_rows)
    if last_time is None:
        return None
    out = dict(source)
    out["task_name"] = "Time_to_Mortality"
    out["source_binary_task"] = "ehrshot_death"
    out["label"] = "tte"
    return add_duration_fields(
        out,
        prediction_time,
        death_time,
        last_time,
        death_time is not None,
        horizon_days,
    )


def ehrshot_death_row_from_patient_rows(
    source: Dict[str, str],
    patient_rows: List[Dict[str, str]],
    horizon_days: float,
) -> Optional[Dict[str, Any]]:
    prediction_time = parse_time(source.get("prediction_time"))
    if prediction_time is None:
        return None
    death_time, last_time = ehrshot_patient_event_times(patient_rows)
    if last_time is None:
        return None
    out = dict(source)
    out["task_name"] = "Time_to_Mortality"
    out["source_binary_task"] = "ehrshot_death"
    out["label"] = "tte"
    return add_duration_fields(
        out,
        prediction_time,
        death_time,
        last_time,
        death_time is not None,
        horizon_days,
    )


def process_ehrshot_record(source):
    return ehrshot_tte_row_from_root(_EHRSHOT_WORKER_ROOT_DIR, source)


def process_ehrshot_death_record(payload):
    source, horizon_days = payload
    return ehrshot_death_row_from_root(
        _EHRSHOT_WORKER_ROOT_DIR,
        source,
        horizon_days,
    )


def process_ehrshot_patient_group(
    payload: Tuple[str, List[Dict[str, str]], bool, float]
) -> List[Optional[Dict[str, Any]]]:
    patient_id, source_records, death_only, death_horizon_days = payload
    patient_rows = load_ehrshot_patient(_EHRSHOT_WORKER_ROOT_DIR, patient_id)
    if death_only:
        return [
            ehrshot_death_row_from_patient_rows(
                source,
                patient_rows,
                death_horizon_days,
            )
            for source in source_records
        ]
    return [
        ehrshot_tte_row_from_patient_rows(source, patient_rows)
        for source in source_records
    ]


def build_ehrshot(args, split: str):
    index_path = args.ehrshot_train_index_path if split == "train" else args.ehrshot_val_index_path
    if not index_path or not os.path.exists(index_path):
        return
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    if args.death_only:
        seen = set()
        source_records = []
        for source in read_csv_records(index_path):
            key = (source.get("patient_id"), source.get("prediction_time"))
            if key in seen:
                continue
            seen.add(key)
            source_records.append(source)
    else:
        source_records = [
            source
            for source in read_csv_records(index_path)
            if source.get("task_name") in EHRSHOT_TTE_TASKS
        ]
    grouped_records = group_records_by_key(
        source_records,
        lambda record: record["patient_id"],
    )
    group_payloads = [
        (
            patient_id,
            records,
            bool(args.death_only),
            float(args.death_horizon_days),
        )
        for patient_id, records in grouped_records
    ]
    worker_count = min(max(1, int(args.num_workers)), max(1, len(group_payloads)))
    progress = tqdm(
        total=len(source_records),
        desc=f"ehrshot {split}",
        unit="sample",
        dynamic_ncols=True,
    )
    try:
        if worker_count <= 1:
            init_ehrshot_worker(args.ehrshot_root_dir)
            iterator = map(process_ehrshot_patient_group, group_payloads)
        else:
            executor = futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=init_ehrshot_worker,
                initargs=(args.ehrshot_root_dir,),
            )
            iterator = executor.map(
                process_ehrshot_patient_group,
                group_payloads,
                chunksize=max(1, int(args.worker_chunksize)),
            )
        for group_rows, payload in zip(iterator, group_payloads):
            progress.update(len(payload[1]))
            for row in group_rows:
                if row is None:
                    continue
                rows_by_task.setdefault(row["task_name"], []).append(row)
    finally:
        progress.close()
        if worker_count > 1 and "executor" in locals():
            executor.shutdown()
    out_dir = os.path.join(args.output_dir, "ehrshot", split)
    for task_name, rows in rows_by_task.items():
        write_csv(os.path.join(out_dir, f"{task_name}.csv"), rows)
        print(f"ehrshot {split} {task_name}: {len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/data/zikun_workspace/tte_task_index")
    parser.add_argument("--mimic_ehr_dir", default="/data/zikun_workspace/mimic-iv-3.1_tabular/patients_ehr")
    parser.add_argument("--mimic_train_index_dir", default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train")
    parser.add_argument("--mimic_val_index_dir", default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val")
    parser.add_argument("--eicu_train_sample_info_path", default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json")
    parser.add_argument("--eicu_val_sample_info_path", default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json")
    parser.add_argument("--eicu_cohorts_path", default="/data/zikun_workspace/eicu-crd/processed/cohorts.csv")
    parser.add_argument("--ehrshot_root_dir", default="/data/EHR_data_public/EHRSHOT")
    parser.add_argument("--ehrshot_train_index_path", default="/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv")
    parser.add_argument("--ehrshot_val_index_path", default="/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--worker_chunksize", type=int, default=32)
    parser.add_argument("--death_only", action="store_true")
    parser.add_argument("--death_horizon_days", type=float, default=3650.0)
    args = parser.parse_args()

    for split in ("train", "val"):
        build_mimic(args, split)
        build_eicu(args, split)
        build_ehrshot(args, split)


if __name__ == "__main__":
    main()
