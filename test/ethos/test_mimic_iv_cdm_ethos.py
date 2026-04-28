import os
import sys
from dataclasses import dataclass, field
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info
from test.ethos.test_ethos_common import parse_args, run_ethos_test


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/EHR_data_public/mimic-iv-cdm")
    task_name: str = field(default="MIMIC-IV-CDM Main Disease Diagnoses")
    max_samples: Optional[int] = field(default=None)
    lazy_mode: bool = field(default=True)
    table_mode: str = field(default="table_only")
    concept_map_dir: Optional[str] = field(default="/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS")


def main():
    args, data_args = parse_args(DataArguments)
    dataset = MIMICIVCDM(
        root_dir=data_args.root_dir,
        split="test",
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode=data_args.table_mode,
        task_name=data_args.task_name,
        max_samples=data_args.max_samples,
        return_meds=True,
        concept_map_dir=data_args.concept_map_dir,
    )
    run_ethos_test(args=args, task_name=data_args.task_name, task_info=get_task_info(), eval_dataset=dataset)


if __name__ == "__main__":
    main()
