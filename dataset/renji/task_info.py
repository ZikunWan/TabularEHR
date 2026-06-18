"""Task metadata for Renji dataset."""

from __future__ import annotations

from copy import deepcopy


TASK_INFO = {
    "single_metric_prediction": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction_template": "Based on the patient's clinical history up to {prediction_point}, predict whether {metric} will be abnormal in the next label window ({label_window}).",
    },
    "multi_label_prediction": {
        "metric": "auroc",
        "task_type": "multi_label_classification",
        "instruction_template": "Based on the patient's clinical history up to {prediction_point}, predict all metrics across all future windows.",
    },
    "tacrolimus_abnormal_survival": {
        "metric": "survival",
        "task_type": "time_to_event",
        "instruction_template": (
            "Based on the patient's clinical history through postoperative day "
            "{prediction_day}, estimate the daily hazard of the first abnormal "
            "tacrolimus concentration during {stage_window}."
        ),
    },
}

ALL_METRICS = sorted(
    [
        "ALT",
        "AST",
        "ALP",
        "GGT",
        "TB",
        "DB",
        "Bile_Acid",
        "TP",
        "ALB",
        "PT",
        "INR",
        "Tacrolimus_Conc",
        "CsA_Trough",
        "CsA_Peak",
        "WBC",
        "N_Percent",
        "Lymphocyte_Abs",
        "HB",
        "PLT",
        "CR",
        "Glucose",
        "Uric_Acid",
        "Triglyceride",
        "Cholesterol",
    ]
)
ALL_POINTS = ["day14", "day30", "day180", "day365", "day395", "day730"]
PREDICTION_POINTS = {
    "day14": (14, "2w-1m", "Day 14"),
    "day30": (30, "2m-6m", "Day 30"),
    "day180": (180, "7m-12m", "Day 180"),
    "day365": (365, "13m-14m", "Day 365"),
    "day395": (395, "15m-24m", "Day 395"),
    "day730": (730, "2y+", "Day 730"),
}


def get_task_info():
    return deepcopy(TASK_INFO)


__all__ = ["TASK_INFO", "ALL_METRICS", "ALL_POINTS", "PREDICTION_POINTS", "get_task_info"]
