import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info
from train_ethos_common import EthosModelArguments, run_training


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/EHR_data_public/eicu-crd/2.0")
    processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    vocab_dir: str = field(default=".cache/ethos_vocab/eicu")
    task_name: str = field(default="mortality")
    train_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_train.json")
    val_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/sample_info_val.json")
    lazy_mode: bool = field(default=True)
    table_mode: str = field(default="table_only")
    max_seq_length: int = field(default=4096)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


def _dataset(data_args, sample_info_path, shuffle, max_samples):
    return EICUDataset(
        root_dir=data_args.root_dir,
        processed_dir=data_args.processed_dir,
        sample_info_path=sample_info_path,
        task_name=data_args.task_name,
        lazy_mode=data_args.lazy_mode,
        shuffle=shuffle,
        table_mode=data_args.table_mode,
        max_samples=max_samples,
        return_meds=True,
    )


def main():
    parser = HfArgumentParser((EthosModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    task_info = get_task_info()
    train_dataset = _dataset(data_args, data_args.train_info_path, True, data_args.max_train_samples)
    eval_dataset = _dataset(data_args, data_args.val_info_path, False, data_args.max_eval_samples)
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
