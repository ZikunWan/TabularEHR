import os
import sys
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dataset.renji.renji_dataset import RenjiDataset
from dataset.renji.task_info import get_task_info
from test.ethos.test_ethos_common import parse_args, run_ethos_test


RENJI_POINTS = ["day0", "day30", "day180", "day365"]


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/EHR_data_public/Renji")
    split: str = field(default="test")
    task_name: str = field(default="multi_label_prediction")
    max_samples: Optional[int] = field(default=None)
    table_mode: str = field(default="text_only")


def main():
    args, data_args = parse_args(DataArguments)
    dataset = RenjiDataset(
        root_dir=data_args.root_dir,
        split=data_args.split,
        max_samples=data_args.max_samples,
        table_mode=data_args.table_mode,
        target_prediction_points=RENJI_POINTS,
        shuffle=False,
        return_meds=True,
    )
    run_ethos_test(args=args, task_name=data_args.task_name, task_info=get_task_info(), eval_dataset=dataset)


if __name__ == "__main__":
    main()
