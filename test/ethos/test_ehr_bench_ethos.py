import os
import sys
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info
from test.ethos.test_ethos_common import parse_args, run_ethos_test


@dataclass
class DataArguments:
    data_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    sample_info_path: Optional[str] = field(default=None)
    task_name: str = field(default="Inpatient_Mortality")
    max_samples: Optional[int] = field(default=None)
    lazy_mode: bool = field(default=True)
    table_mode: str = field(default="table_only")
    itemid_representation: str = field(default="code")
    concept_map_dir: Optional[str] = field(default=None)


def main():
    args, data_args = parse_args(DataArguments)
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    sample_info_path = data_args.sample_info_path or os.path.join(data_args.data_dir, "task_index", "test", f"{data_args.task_name}.csv")
    dataset = MIMICIV(
        root_dir=data_args.data_dir,
        sample_info_path=sample_info_path,
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode=data_args.table_mode,
        max_samples=data_args.max_samples,
        itemid_representation=data_args.itemid_representation,
        concept_map_dir=data_args.concept_map_dir,
        return_meds=True,
    )
    run_ethos_test(args=args, task_name=data_args.task_name, task_info=get_task_info(), eval_dataset=dataset)


if __name__ == "__main__":
    main()
