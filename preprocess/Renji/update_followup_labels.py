#!/usr/bin/env python3
"""Rebuild all follow-up *_label columns from cleaned Renji follow-up CSV files."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_ROOT_DIR = "/data/EHR_data_public/Renji"
DEFAULT_FOLLOWUP_DIR = os.path.join(DEFAULT_ROOT_DIR, "follow_ups")
DEFAULT_ENCODING = "utf-8-sig"
PATIENT_INFO_FILENAME = "患儿基本信息总表251023_含免疫事件.xlsx"
DEFAULT_AGE_YEARS = 5.0

ABNORMAL_FLAGS = {"Abnormal", "Elevated", "Very High"}
NORMAL_FLAGS = {"Normal"}

DRUG_CONC_MED_COLS = {
    "Tacrolimus_Conc": ["他克莫司缓释胶囊", "他克莫司(赛福开)", "他克莫司(普乐可复)"],
}

DRUG_CONC_RANGES = {
    "Tacrolimus_Conc": [
        (0, 31, 8.0, 12.0),
        (31, 181, 7.0, 10.0),
        (181, 366, 5.0, 8.0),
        (366, float("inf"), 4.0, 6.0),
    ],
}

ZH_TO_EN = {
    "WBC": "WBC",
    "N(%)": "N_Percent",
    "淋巴细胞绝对值": "Lymphocyte_Abs",
    "嗜酸性粒细胞百分比": "Eosinophil_Percent",
    "HB": "HB",
    "PLT": "PLT",
    "TP": "TP",
    "ALB": "ALB",
    "ALT": "ALT",
    "AST": "AST",
    "ALP": "ALP",
    "γ-GT": "GGT",
    "DB": "DB",
    "TB": "TB",
    "胆汁酸": "Bile_Acid",
    "CR": "CR",
    "血糖": "Glucose",
    "甘油三脂": "Triglyceride",
    "总胆固醇": "Cholesterol",
    "尿酸": "Uric_Acid",
    "PT": "PT",
    "INR": "INR",
    "雷帕浓度": "Rapa_Conc",
    "血氨": "Blood_Ammonia",
    "他克莫司浓度": "Tacrolimus_Conc",
    "环孢素谷浓度": "CsA_Trough",
    "环孢素峰浓度": "CsA_Peak",
    "CMV-DNA": "CMV_DNA",
    "EBV-DNA": "EBV_DNA",
    "HBV-DNA": "HBV_DNA",
    "HBsAg": "HBsAg",
    "HBsAb": "HBsAb",
    "HBeAg": "HBeAg",
    "HBeAb": "HBeAb",
    "HBcAb": "HBcAb",
}

LAB_META = {
    "WBC": {
        "unit": "10^9/L",
        "ranges": [
            (0, 0.5, None, 4.3, 14.2),
            (0.5, 1, None, 4.8, 14.6),
            (1, 2, None, 5.1, 14.1),
            (2, 6, None, 4.4, 11.9),
            (6, 13, None, 4.3, 11.3),
            (13, 99, None, 4.1, 11.0),
        ],
    },
    "N_Percent": {
        "unit": "%",
        "ranges": [
            (0, 0.5, None, 7, 56),
            (0.5, 1, None, 9, 57),
            (1, 2, None, 13, 55),
            (2, 6, None, 22, 65),
            (6, 13, None, 31, 70),
            (13, 99, None, 37, 77),
        ],
    },
    "Lymphocyte_Abs": {
        "unit": "10^9/L",
        "ranges": [
            (0, 0.5, None, 2.4, 9.5),
            (0.5, 1, None, 2.5, 9.0),
            (1, 2, None, 2.4, 8.7),
            (2, 6, None, 1.8, 6.3),
            (6, 13, None, 1.5, 4.6),
            (13, 99, None, 1.2, 3.8),
        ],
    },
    "Eosinophil_Percent": {
        "unit": "%",
        "ranges": [(0, 2, None, 0.0, 0.1), (2, 99, None, 0.0, 0.07)],
    },
    "HB": {
        "unit": "g/L",
        "ranges": [
            (0, 0.5, None, 97, 183),
            (0.5, 1, None, 97, 141),
            (1, 2, None, 107, 141),
            (2, 6, None, 112, 149),
            (6, 13, None, 118, 156),
            (13, 99, "M", 129, 172),
            (13, 99, "F", 114, 154),
        ],
    },
    "PLT": {
        "unit": "10^9/L",
        "ranges": [
            (0, 0.5, None, 183, 614),
            (0.5, 1, None, 190, 445),
            (1, 2, None, 190, 472),
            (2, 6, None, 188, 472),
            (6, 13, None, 167, 453),
            (13, 18, None, 150, 407),
        ],
    },
    "TP": {
        "unit": "g/L",
        "ranges": [
            (0, 0.5, None, 49, 71),
            (0.5, 1, None, 55, 75),
            (1, 2, None, 58, 76),
            (2, 6, None, 61, 79),
            (6, 13, None, 65, 84),
            (13, 99, None, 68, 88),
        ],
    },
    "ALB": {
        "unit": "g/L",
        "ranges": [(0, 0.5, None, 35, 50), (0.5, 13, None, 39, 54), (13, 99, None, 42, 56)],
    },
    "ALT": {
        "unit": "U/L",
        "ranges": [
            (0, 1, None, 8, 71),
            (1, 2, None, 8, 42),
            (2, 13, None, 7, 30),
            (13, 99, "M", 7, 43),
            (13, 99, "F", 6, 29),
        ],
    },
    "AST": {
        "unit": "U/L",
        "ranges": [
            (0, 1, None, 21, 80),
            (1, 2, None, 22, 59),
            (2, 13, None, 14, 44),
            (13, 99, "M", 12, 37),
            (13, 99, "F", 10, 31),
        ],
    },
    "ALP": {
        "unit": "U/L",
        "ranges": [
            (0, 0.5, None, 98, 532),
            (0.5, 1, None, 106, 420),
            (1, 2, None, 128, 432),
            (2, 9, None, 143, 406),
            (9, 12, None, 146, 500),
            (12, 14, "M", 160, 610),
            (12, 14, "F", 81, 454),
            (14, 15, "M", 82, 603),
            (14, 15, "F", 63, 327),
            (15, 17, "M", 64, 443),
            (15, 17, "F", 52, 215),
            (17, 99, "M", 51, 202),
            (17, 99, "F", 43, 130),
        ],
    },
    "GGT": {
        "unit": "U/L",
        "ranges": [
            (0, 0.5, None, 9, 150),
            (0.5, 1, None, 6, 31),
            (1, 13, None, 5, 19),
            (13, 99, "M", 8, 40),
            (13, 99, "F", 6, 26),
        ],
    },
    "DB": {"unit": "μmol/L", "ranges": [(0, 99, None, 0, 6.84)]},
    "TB": {"unit": "μmol/L", "ranges": [(0, 99, None, 0, 23)]},
    "Bile_Acid": {"unit": "μmol/L", "ranges": [(0, 99, None, 0.01, 10)]},
    "CR": {
        "unit": "μmol/L",
        "ranges": [
            (0, 2, None, 13, 33),
            (2, 6, None, 19, 44),
            (6, 13, None, 27, 66),
            (13, 16, "M", 37, 93),
            (13, 16, "F", 33, 75),
            (16, 99, "M", 52, 101),
            (16, 99, "F", 39, 76),
        ],
    },
    "Glucose": {"unit": "mmol/L", "ranges": [(0, 99, None, 3.9, 6.1)]},
    "Triglyceride": {
        "unit": "mmol/L",
        "ranges": [(0, 99, None, 0, 1.7)],
        "severity_bands": [
            {"label": "Normal", "range": (0, 99, None, 0, 1.7)},
            {"label": "Elevated", "range": (0, 99, None, 1.7, 2.3)},
            {"label": "Very High", "range": (0, 99, None, 2.3, float("inf"))},
        ],
    },
    "Cholesterol": {
        "unit": "mmol/L",
        "ranges": [(0, 99, None, 0, 5.2)],
        "severity_bands": [
            {"label": "Normal", "range": (0, 99, None, 0, 5.2)},
            {"label": "Elevated", "range": (0, 99, None, 5.2, 6.2)},
            {"label": "Very High", "range": (0, 99, None, 6.2, float("inf"))},
        ],
    },
    "Uric_Acid": {"unit": "μmol/L", "ranges": [(0, 99, None, 155, 428)]},
    "PT": {"unit": "s", "ranges": [(0, 99, None, 9.4, 12.5)]},
    "INR": {"unit": "", "ranges": [(0, 99, None, 0.8, 1.15)]},
    "Blood_Ammonia": {"unit": "μmol/L", "ranges": [(0, 99, None, 9, 30)]},
    "Tacrolimus_Conc": {"unit": "ng/mL", "ranges": []},
    "CsA_Trough": {"unit": "ng/mL", "ranges": [(0, 99, None, 100, 400)]},
    "CsA_Peak": {"unit": "ng/mL", "ranges": [(0, 99, None, 400, 1600)]},
    "Rapa_Conc": {"unit": "ng/mL", "ranges": [(0, 99, None, 5, 20)]},
    "CMV_DNA": {"unit": "copies/mL", "ranges": [(0, 99, None, 0, 400)]},
    "EBV_DNA": {"unit": "copies/mL", "ranges": [(0, 99, None, 0, 400)]},
    "HBV_DNA": {"unit": "IU/mL", "ranges": [(0, 99, None, 0, 20)]},
    "HBsAg": {"unit": "COI", "ranges": [(0, 99, None, 0, 1)]},
    "HBsAb": {"unit": "mIU/mL", "ranges": [(0, 99, None, 0, 10)]},
    "HBeAg": {"unit": "COI", "ranges": [(0, 99, None, 0, 1)]},
    "HBeAb": {"unit": "COI", "ranges": [(0, 99, None, 1, float("inf"))]},
    "HBcAb": {"unit": "COI", "ranges": [(0, 99, None, 1, float("inf"))]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute all *_label columns in Renji follow-up CSV files."
    )
    parser.add_argument(
        "--root-dir",
        default=DEFAULT_ROOT_DIR,
        help="Renji root directory containing patient info.",
    )
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLLOWUP_DIR,
        help="Directory containing follow-up CSV files to update.",
    )
    parser.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help="Preferred CSV encoding.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute changes without writing files.",
    )
    return parser.parse_args()


def read_csv_rows(file_path: str, encoding: str) -> Tuple[List[str], List[List[str]]]:
    encodings = [encoding]
    for extra in ("utf-8", "utf-8-sig", "gb18030"):
        if extra not in encodings:
            encodings.append(extra)

    last_error = None
    for current_encoding in encodings:
        try:
            with open(file_path, "r", encoding=current_encoding, newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return [], []
            return rows[0], rows[1:]
        except Exception as exc:
            last_error = exc
    raise last_error


def load_patient_info_map(root_dir: str) -> Dict[str, pd.Series]:
    patient_info_path = os.path.join(root_dir, PATIENT_INFO_FILENAME)
    if not os.path.exists(patient_info_path):
        return {}

    patient_info_df = pd.read_excel(patient_info_path)
    patient_info_map = {}
    for _, row in patient_info_df.iterrows():
        tid = str(row.get("transplant_id", "")).strip()
        key = tid.rsplit("_", 1)[0] if "_" in tid else tid
        if key:
            patient_info_map[key] = row
    return patient_info_map


def infer_gender(patient_row) -> Optional[str]:
    if patient_row is None:
        return None

    recipient_gender = patient_row.get("recipient_gender")
    if pd.isna(recipient_gender):
        return None

    gender_str = str(recipient_gender).strip().upper()
    if gender_str in {"M", "MALE", "男"}:
        return "M"
    if gender_str in {"F", "FEMALE", "女"}:
        return "F"
    return None


def infer_age_years(patient_row, report_date) -> float:
    if patient_row is None or report_date is None or pd.isna(report_date):
        return DEFAULT_AGE_YEARS

    dob = patient_row.get("date_of_birth")
    if pd.isna(dob):
        return DEFAULT_AGE_YEARS

    try:
        dob_ts = pd.to_datetime(dob, errors="coerce")
        report_ts = pd.to_datetime(report_date, errors="coerce")
        if pd.isna(dob_ts) or pd.isna(report_ts):
            return DEFAULT_AGE_YEARS
        return max((report_ts - dob_ts).days / 365.25, 0.0)
    except Exception:
        return DEFAULT_AGE_YEARS


def get_patient_info_row(patient_info_map: Dict[str, pd.Series], file_name: str):
    fname_key = os.path.splitext(file_name)[0]
    return patient_info_map.get(fname_key)


def get_metric_columns(header: List[str]) -> List[Tuple[str, str, str]]:
    metric_columns = []
    for col in header:
        if not col or col.endswith("_label"):
            continue
        metric_en = ZH_TO_EN.get(col)
        if metric_en and metric_en in LAB_META:
            metric_columns.append((col, metric_en, f"{col}_label"))
    return metric_columns


def get_reference_range(lab_item: str, age_years: float, gender: Optional[str]) -> Tuple[Optional[float], Optional[float], str]:
    if lab_item not in LAB_META:
        return None, None, ""

    meta = LAB_META[lab_item]
    unit = meta["unit"]
    ranges = meta["ranges"]

    for age_min, age_max, range_gender, low, high in ranges:
        if age_min <= age_years < age_max:
            if range_gender is None or range_gender == gender:
                return low, high, unit

    for age_min, age_max, range_gender, low, high in ranges:
        if age_min <= age_years < age_max and range_gender is None:
            return low, high, unit

    if ranges:
        _, _, _, low, high = ranges[-1]
        return low, high, unit
    return None, None, unit


def is_blank(value) -> bool:
    if value is None:
        return True
    value_str = str(value).strip()
    return value_str == "" or value_str.lower() in {"nan", "none", "null"}


def numeric_values(value) -> List[float]:
    if is_blank(value):
        return []
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]


def parse_postop_day(value) -> Optional[float]:
    if is_blank(value):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def row_value(row: List[str], header_index: Dict[str, int], col: str) -> str:
    idx = header_index.get(col)
    if idx is None or idx >= len(row):
        return ""
    return row[idx]


def get_first_drug_days(header_index: Dict[str, int], rows: List[List[str]]) -> Dict[str, Optional[float]]:
    postop_idx = header_index.get("术后天数")
    first_drug_days = {}

    for lab_item, med_cols in DRUG_CONC_MED_COLS.items():
        med_days = []
        conc_days = []
        conc_col = next((col for col, metric_en in ZH_TO_EN.items() if metric_en == lab_item), None)

        for row in rows:
            day = parse_postop_day(row[postop_idx]) if postop_idx is not None and postop_idx < len(row) else None
            if day is None:
                continue

            has_drug = any(
                any(value > 0 for value in numeric_values(row_value(row, header_index, med_col)))
                for med_col in med_cols
            )
            if has_drug:
                med_days.append(day)

            if conc_col and not is_blank(row_value(row, header_index, conc_col)):
                conc_days.append(day)

        first_days = med_days + conc_days
        first_drug_days[lab_item] = min(first_days) if first_days else None

    return first_drug_days


def get_drug_concentration_reference_range(
    lab_item: str,
    postop_day: Optional[float],
    first_drug_days: Optional[Dict[str, Optional[float]]],
) -> Tuple[Optional[float], Optional[float], str]:
    unit = LAB_META.get(lab_item, {}).get("unit", "")
    first_day = (first_drug_days or {}).get(lab_item)
    if postop_day is None or first_day is None:
        return None, None, unit

    elapsed_days = postop_day - first_day
    for day_min, day_max, low, high in DRUG_CONC_RANGES[lab_item]:
        if day_min <= elapsed_days < day_max:
            return low, high, unit

    return None, None, unit


def get_severity_flag(lab_item: str, value: float, age_years: float, gender: Optional[str]) -> Optional[str]:
    meta = LAB_META.get(lab_item, {})
    severity_bands = meta.get("severity_bands")
    if not severity_bands:
        return None

    for band in severity_bands:
        band_range = band.get("range")
        label = band.get("label")
        if not band_range or not label or len(band_range) != 5:
            continue

        age_min, age_max, range_gender, low, high = band_range
        if not (age_min <= age_years < age_max):
            continue
        if range_gender is not None and range_gender != gender:
            continue

        if high == float("inf"):
            in_range = value >= low
        else:
            in_range = low <= value < high

        if in_range:
            return label

    return None


def split_multivalue_parts(value: str) -> List[str]:
    value_str = str(value).strip()
    if not value_str or "-" not in value_str:
        return [value_str]

    raw_parts = [part.strip() for part in value_str.split("-") if part.strip()]
    if len(raw_parts) < 2:
        return [value_str]

    def is_meaningful_part(part: str) -> bool:
        part_lower = part.lower()
        if part in {"阴性", "阳性"} or part_lower in {"negative", "positive", "normal", "abnormal"}:
            return True
        if any(marker in part for marker in ("<", ">", "≤", "≥")):
            return True
        try:
            float(part)
            return True
        except Exception:
            return False

    if all(is_meaningful_part(part) for part in raw_parts):
        return raw_parts
    return [value_str]


def describe_lab_value(
    lab_item: str,
    value: str,
    age_years: float,
    gender: Optional[str],
    postop_day: Optional[float] = None,
    first_drug_days: Optional[Dict[str, Optional[float]]] = None,
) -> Dict[str, str]:
    if lab_item in DRUG_CONC_RANGES:
        low_limit, high_limit, unit = get_drug_concentration_reference_range(
            lab_item,
            postop_day,
            first_drug_days,
        )
    else:
        low_limit, high_limit, unit = get_reference_range(lab_item, age_years, gender)

    raw_value = str(value).strip()
    raw_lower = raw_value.lower()

    if raw_value == "" or raw_lower in {"nan", "none", "null"}:
        return {"flag": "Unknown", "unit": unit, "value": raw_value}

    if raw_value == "阴性" or raw_lower == "negative":
        return {"flag": "Normal", "unit": "", "value": "Normal"}

    if raw_value == "阳性" or raw_lower == "positive":
        return {"flag": "Abnormal", "unit": "", "value": "Abnormal"}

    if "<" in raw_value or "≤" in raw_value:
        return {"flag": "Normal", "unit": unit, "value": raw_value}

    if ">" in raw_value or "≥" in raw_value:
        return {"flag": "Abnormal", "unit": unit, "value": raw_value}

    try:
        numeric_value = float(raw_value)
        severity_flag = get_severity_flag(lab_item, numeric_value, age_years, gender)
        if severity_flag is not None:
            return {"flag": severity_flag, "unit": unit, "value": raw_value}
        if low_limit is not None and high_limit is not None:
            flag = "Abnormal" if (numeric_value < low_limit or numeric_value > high_limit) else "Normal"
            return {"flag": flag, "unit": unit, "value": raw_value}
    except Exception:
        pass

    return {"flag": "Unknown", "unit": unit, "value": raw_value}


def compute_label_for_value(
    metric_en: str,
    raw_value: str,
    age_years: float,
    gender: Optional[str],
    postop_day: Optional[float] = None,
    first_drug_days: Optional[Dict[str, Optional[float]]] = None,
) -> str:
    if raw_value is None:
        return ""

    value_str = str(raw_value).strip()
    if not value_str or value_str.lower() in {"nan", "none", "null"}:
        return ""

    parts = split_multivalue_parts(value_str)
    saw_normal = False
    saw_unknown = False

    for part in parts:
        desc = describe_lab_value(metric_en, part, age_years, gender, postop_day, first_drug_days)
        flag = str(desc.get("flag", "")).strip()

        if flag in ABNORMAL_FLAGS:
            return "1"
        if flag in NORMAL_FLAGS:
            saw_normal = True
        else:
            saw_unknown = True

    if saw_normal:
        return "0"
    if saw_unknown:
        return ""
    return ""


def rebuild_file_labels(
    file_path: str,
    patient_info_map: Dict[str, pd.Series],
    encoding: str,
    dry_run: bool,
) -> Dict[str, int]:
    header, data_rows = read_csv_rows(file_path, encoding)
    if not header:
        return {"files_changed": 0, "label_cells_updated": 0, "label_cols_added": 0}

    patient_row = get_patient_info_row(patient_info_map, os.path.basename(file_path))
    gender = infer_gender(patient_row)

    header_index = {col: idx for idx, col in enumerate(header)}
    metric_columns = get_metric_columns(header)
    first_drug_days = get_first_drug_days(header_index, data_rows)

    label_cols_added = 0
    for source_col, _, label_col in metric_columns:
        if label_col not in header_index:
            header_index[label_col] = len(header)
            header.append(label_col)
            label_cols_added += 1
            for row in data_rows:
                row.append("")

    report_date_idx = header_index.get("报告日期")
    postop_day_idx = header_index.get("术后天数")

    changed = False
    label_cells_updated = 0

    for row in data_rows:
        if len(row) < len(header):
            row.extend([""] * (len(header) - len(row)))

        report_date = row[report_date_idx] if report_date_idx is not None and report_date_idx < len(row) else None
        postop_day = row[postop_day_idx] if postop_day_idx is not None and postop_day_idx < len(row) else None
        age_years = infer_age_years(patient_row, report_date)

        for source_col, metric_en, label_col in metric_columns:
            src_idx = header_index[source_col]
            label_idx = header_index[label_col]
            raw_value = row[src_idx] if src_idx < len(row) else ""
            new_label = compute_label_for_value(
                metric_en,
                raw_value,
                age_years,
                gender,
                parse_postop_day(postop_day),
                first_drug_days,
            )
            old_label = row[label_idx].strip() if label_idx < len(row) else ""

            if new_label != old_label:
                row[label_idx] = new_label
                changed = True
                label_cells_updated += 1

    if changed and not dry_run:
        fd, tmp_path = tempfile.mkstemp(prefix="renji_labels_", suffix=".csv", dir=os.path.dirname(file_path))
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding=encoding, newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data_rows)
            os.replace(tmp_path, file_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return {
        "files_changed": 1 if changed else 0,
        "label_cells_updated": label_cells_updated,
        "label_cols_added": label_cols_added,
    }


def main():
    args = parse_args()

    if not os.path.isdir(args.folder):
        raise SystemExit(f"Follow-up directory not found: {args.folder}")

    patient_info_map = load_patient_info_map(args.root_dir)
    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(".csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {args.folder}")

    total_files_changed = 0
    total_label_cells_updated = 0
    total_label_cols_added = 0

    for idx, file_name in enumerate(files, 1):
        file_path = os.path.join(args.folder, file_name)
        stats = rebuild_file_labels(file_path, patient_info_map, args.encoding, args.dry_run)
        total_files_changed += stats["files_changed"]
        total_label_cells_updated += stats["label_cells_updated"]
        total_label_cols_added += stats["label_cols_added"]

        if idx % 200 == 0 or idx == len(files):
            print(
                f"[{idx}/{len(files)}] files_changed={total_files_changed}, "
                f"label_cells_updated={total_label_cells_updated}"
            )

    mode = "DRY-RUN" if args.dry_run else "DONE"
    print(
        f"{mode}: scanned={len(files)}, files_changed={total_files_changed}, "
        f"label_cells_updated={total_label_cells_updated}, "
        f"label_cols_added={total_label_cols_added}"
    )


if __name__ == "__main__":
    main()
