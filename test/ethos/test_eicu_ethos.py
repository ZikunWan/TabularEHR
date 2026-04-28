import os
import sys
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info
from test.ethos.test_ethos_common import parse_args, run_ethos_test


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/EHR_data_public/eicu-crd/2.0")
    processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    sample_info_test_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_test.json")
    task_name: str = field(default="mortality")
    max_samples: Optional[int] = field(default=None)
    lazy_mode: bool = field(default=True)
    table_mode: str = field(default="table_only")


def main():
    args, data_args = parse_args(DataArguments)
    dataset = EICUDataset(
        root_dir=data_args.root_dir,
        processed_dir=data_args.processed_dir,
        sample_info_path=data_args.sample_info_test_path,
        task_name=data_args.task_name,
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode=data_args.table_mode,
        max_samples=data_args.max_samples,
        return_meds=True,
    )
    run_ethos_test(args=args, task_name=data_args.task_name, task_info=get_task_info(), eval_dataset=dataset)


if __name__ == "__main__":
    main()
