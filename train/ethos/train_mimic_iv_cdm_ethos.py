import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info
from train_ethos_common import EthosModelArguments, run_training


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/EHR_data_public/mimic-iv-cdm")
    vocab_dir: str = field(default=".cache/ethos_vocab/mimic_iv_cdm/main_disease")
    task_name: str = field(default="MIMIC-IV-CDM Main Disease Diagnoses")
    lazy_mode: bool = field(default=True)
    table_mode: str = field(default="table_only")
    concept_map_dir: Optional[str] = field(default="/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS")
    max_seq_length: int = field(default=2048)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


def _dataset(data_args, split, shuffle, max_samples):
    return MIMICIVCDM(
        root_dir=data_args.root_dir,
        split=split,
        lazy_mode=data_args.lazy_mode,
        shuffle=shuffle,
        table_mode=data_args.table_mode,
        task_name=data_args.task_name,
        max_samples=max_samples,
        return_meds=True,
        concept_map_dir=data_args.concept_map_dir,
    )


def main():
    parser = HfArgumentParser((EthosModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    task_info = get_task_info()
    train_dataset = _dataset(data_args, "train", True, data_args.max_train_samples)
    eval_dataset = _dataset(data_args, "val", False, data_args.max_eval_samples)
    run_training(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        task_name=data_args.task_name,
        task_info=task_info,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


if __name__ == "__main__":
    main()
