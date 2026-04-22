"""Task metadata for EHRSHOT."""
from __future__ import annotations
from copy import deepcopy


TASK_INFO = {
    "guo_los": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether a patient's total length of stay during a visit to the hospital will be at least 7 days.",
    },
    "guo_readmission": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether a patient will be re-admitted to the hospital within 30 days after being discharged from a visit.",
    },
    "guo_icu": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether a patient will be transferred to the ICU during a visit to the hospital.",
    },
    "lab_anemia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming anemia lab result as 0 (Normal, >=120 g/L), 1 (Mild, 110-119 g/L), 2 (Moderate, 80-109 g/L), or 3 (Severe, <80 g/L).",
    },
    "lab_hyperkalemia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming hyperkalemia lab result as 0 (Normal, <=5.5 mmol/L), 1 (Mild, >5.5 and <=6 mmol/L), 2 (Moderate, >6 and <=7 mmol/L), or 3 (Severe, >7 mmol/L).",
    },
    "lab_hyponatremia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming hyponatremia lab result as 0 (Normal, >=135 mmol/L), 1 (Mild, 130-134 mmol/L), 2 (Moderate, 125-129 mmol/L), or 3 (Severe, <125 mmol/L).",
    },
    "lab_hypoglycemia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming hypoglycemia lab result as 0 (Normal, >=3.9 mmol/L), 1 (Mild, 3.5-3.8 mmol/L), 2 (Moderate, 3.0-3.4 mmol/L), or 3 (Severe, <3.0 mmol/L).",
    },
    "lab_thrombocytopenia": {
        "metric": "auroc",
        "task_type": "multi_class_classification",
        "num_classes": 4,
        "instruction": "Given the sequence of events that have occurred in a hospital, please classify the severity of the upcoming thrombocytopenia lab result as 0 (Normal, >=150 10^9/L), 1 (Mild, 100-149 10^9/L), 2 (Moderate, 50-99 10^9/L), or 3 (Severe, <50 10^9/L).",
    },
    "new_acutemi": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of an acute myocardial infarction within the next year.",
    },
    "new_celiac": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of celiac disease within the next year.",
    },
    "new_hyperlipidemia": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of hyperlipidemia within the next year.",
    },
    "new_hypertension": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of essential hypertension within the next year.",
    },
    "new_lupus": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of lupus within the next year.",
    },
    "new_pancan": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of pancreatic cancer within the next year.",
    },
}


def get_task_info():
    return deepcopy(TASK_INFO)


__all__ = ["TASK_INFO", "get_task_info"]
