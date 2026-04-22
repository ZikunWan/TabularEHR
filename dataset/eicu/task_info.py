"""Task metadata for eICU."""
from __future__ import annotations
from copy import deepcopy


TASK_INFO = {
    "mortality": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of ICU events observed in the first 12 hours of the ICU stay, predict whether the patient will die during or shortly after ICU stay in the next 24 hours.",
    },

    "long_term_mortality": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of ICU events observed in the first 12 hours of the ICU stay, predict whether the patient will die within the next 14 days.",
    },

    "readmission": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of ICU events, predict whether the patient will be readmitted to ICU after discharge from the current stay.",
    },

    "los_3day": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of ICU events, predict whether the patient's total ICU length of stay will exceed 3 days.",
    },

    "los_7day": {
        "metric": "auroc",
        "task_type": "binary_classification",
        "instruction": "Given the sequence of ICU events, predict whether the patient's total ICU length of stay will exceed 7 days.",
    },

    "final_acuity": {
        "metric": "accuracy",
        "task_type": "multi_class_classification",
        "num_classes": 6,
        "candidate": ['Home', 'Rehabilitation', 'Skilled Nursing Facility', 'Other', 'IN_ICU_MORTALITY', 'IN_HOSPITAL_MORTALITY'],
        "instruction": "Given the sequence of ICU events, predict the patient's final acuity outcome (e.g., Home, Death, Skilled Nursing Facility, Rehabilitation, etc.).",
    },

    "imminent_discharge": {
        "metric": "accuracy",
        "task_type": "multi_class_classification",
        "num_classes": 6,
        "candidate": ['No discharge', 'Death', 'Home', 'Rehabilitation', 'Skilled Nursing Facility', 'Other'],
        "instruction": "Given the sequence of ICU events observed in the first 12 hours of the ICU stay, predict whether and where the patient will be discharged in the next 24 hours.",
    },

    "diagnosis": {
        "metric": "recall",
        "task_type": "multi_label_classification",
        "num_classes": 17,
        "caption": "CCS LVL 1 contains 18 diagnostic classes; however, class 14 was removed as it represents a rare class with a negligible footprint in the dataset.",
        "instruction": "Given the sequence of ICU events, predict which CCS (Clinical Classifications Software) disease categories the patient will be diagnosed with. This is a multi-label task.",
    },

    "creatinine": {
        "metric": "accuracy",
        "task_type": "multi_class_classification",
        "num_classes": 5,
        "instruction": "Given the sequence of ICU events observed in the first 12 hours of the ICU stay, predict the creatinine severity level (0-4) in the next 24 hours based on SOFA criteria: 0:<1.2, 1:1.2-2.0, 2:2.0-3.5, 3:3.5-5.0, 4:>=5.0 mg/dL.",
    },

    "bilirubin": {
        "metric": "accuracy",
        "task_type": "multi_class_classification",
        "num_classes": 5,
        "instruction": "Given the sequence of ICU events observed in the first 12 hours of the ICU stay, predict the bilirubin severity level (0-4) in the next 24 hours based on SOFA criteria: 0:<1.2, 1:1.2-2.0, 2:2.0-6.0, 3:6.0-12.0, 4:>=12.0 mg/dL.",
    },

    "platelets": {
        "metric": "accuracy",
        "task_type": "multi_class_classification",
        "num_classes": 5,
        "instruction": "Given the sequence of ICU events observed in the first 12 hours of the ICU stay, predict the platelet count severity level (0-4) in the next 24 hours based on SOFA criteria: 0:>=150, 1:100-150, 2:50-100, 3:20-50, 4:<20 x10^3/uL.",
    },

    "wbc": {
        "metric": "accuracy",
        "task_type": "multi_class_classification",
        "num_classes": 3,
        "instruction": "Given the sequence of ICU events observed in the first 12 hours of the ICU stay, predict the WBC count severity level (0-2) in the next 24 hours: 0:<4, 1:4-12, 2:>12 x10^3/uL.",
    },
}


def get_task_info():
    return deepcopy(TASK_INFO)


__all__ = ["TASK_INFO", "get_task_info"]
