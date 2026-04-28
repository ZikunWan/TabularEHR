import os
import sys
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info
from test.ethos.test_ethos_common import parse_args, run_ethos_test


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/EHR_data_public/EHRSHOT")
    test_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/index/ehrshot_test.csv")
    task_name: str = field(default="lab_anemia")
    max_samples: Optional[int] = field(default=None)
    lazy_mode: bool = field(default=True)
    table_mode: str = field(default="table_only")


def main():
    args, data_args = parse_args(DataArguments)
    dataset = EHRSHOTDataset(
        root_dir=data_args.root_dir,
        sample_info_path=data_args.test_info_path,
        task_name=data_args.task_name,
        lazy_mode=data_args.lazy_mode,
        table_mode=data_args.table_mode,
        max_samples=data_args.max_samples,
        return_meds=True,
    )
    run_ethos_test(args=args, task_name=data_args.task_name, task_info=get_task_info(), eval_dataset=dataset)


if __name__ == "__main__":
    main()
