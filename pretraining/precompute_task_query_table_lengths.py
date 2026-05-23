import os
import sys
from dataclasses import dataclass, field
from typing import List

from transformers import HfArgumentParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info as get_ehrshot_task_info
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info as get_eicu_task_info
from dataset.mimic.mimic_dataset import MIMICIV


RISK_PREDICTION_TASKS = [
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


@dataclass
class DataArguments:
    dataset: List[str] = field(default_factory=lambda: ["mimic_iv", "eicu", "ehrshot"])
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    train_sample_info_path: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train")
    val_sample_info_path: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val")
    eicu_root_dir: str = field(default="/data/zikun_workspace/eicu-crd")
    eicu_processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    eicu_train_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json")
    eicu_val_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json")
    ehrshot_root_dir: str = field(default="/data/EHR_data_public/EHRSHOT")
    ehrshot_train_sample_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv")
    ehrshot_val_sample_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv")


def resolve_sample_info_paths(path_arg: str):
    paths = []
    for raw_path in path_arg.split(","):
        path = raw_path.strip()
        if not path:
            continue
        if os.path.isdir(path):
            for task_name in RISK_PREDICTION_TASKS:
                csv_path = os.path.join(path, f"{task_name}.csv")
                if os.path.exists(csv_path):
                    paths.append(csv_path)
        else:
            paths.append(path)
    return paths


def binary_task_names(task_info: dict):
    return sorted(
        task_name
        for task_name, info in task_info.items()
        if info["task_type"] == "binary_classification"
    )


def precompute_mimic(data_args: DataArguments):
    for sample_info_path in resolve_sample_info_paths(data_args.train_sample_info_path) + resolve_sample_info_paths(data_args.val_sample_info_path):
        MIMICIV(
            root_dir=data_args.root_dir,
            sample_info_path=sample_info_path,
            lazy_mode=True,
            shuffle=False,
            table_mode="table_only",
            max_samples=None,
            use_table_length_cache=True,
        )


def precompute_eicu(data_args: DataArguments):
    task_names = binary_task_names(get_eicu_task_info())
    for sample_info_path in [data_args.eicu_train_sample_info_path, data_args.eicu_val_sample_info_path]:
        for task_name in task_names:
            EICUDataset(
                root_dir=data_args.eicu_root_dir,
                processed_dir=data_args.eicu_processed_dir,
                sample_info_path=sample_info_path,
                task_name=task_name,
                lazy_mode=True,
                shuffle=False,
                table_mode="table_only",
                use_table_length_cache=True,
            )


def precompute_ehrshot(data_args: DataArguments):
    task_names = binary_task_names(get_ehrshot_task_info())
    for sample_info_path in [data_args.ehrshot_train_sample_info_path, data_args.ehrshot_val_sample_info_path]:
        for task_name in task_names:
            EHRSHOTDataset(
                root_dir=data_args.ehrshot_root_dir,
                sample_info_path=sample_info_path,
                task_name=task_name,
                lazy_mode=True,
                table_mode="table_only",
                use_table_length_cache=True,
            )


def main():
    parser = HfArgumentParser((DataArguments,))
    (data_args,) = parser.parse_args_into_dataclasses()

    if "mimic_iv" in data_args.dataset:
        precompute_mimic(data_args)
    if "eicu" in data_args.dataset:
        precompute_eicu(data_args)
    if "ehrshot" in data_args.dataset:
        precompute_ehrshot(data_args)


if __name__ == "__main__":
    main()
