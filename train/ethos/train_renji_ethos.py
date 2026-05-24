import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

from dataset.renji.renji_dataset import RenjiDataset
from dataset.renji.task_info import get_task_info
from train_ethos_common import EthosModelArguments, run_training


RENJI_POINTS = ["day0", "day30", "day180", "day365"]


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/EHR_data_public/Renji")
    vocab_dir: str = field(default=".cache/ethos_vocab/renji")
    task_name: str = field(default="multi_label_prediction")
    eval_split: str = field(default="all_valid")
    table_mode: str = field(default="text_only")
    max_seq_length: int = field(default=4096)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


def _dataset(data_args, split, shuffle, max_samples):
    return RenjiDataset(
        root_dir=data_args.root_dir,
        split=split,
        max_samples=max_samples,
        table_mode=data_args.table_mode,
        target_prediction_points=RENJI_POINTS,
        shuffle=shuffle,
        return_meds=True,
    )


def main():
    parser = HfArgumentParser((EthosModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    task_info = get_task_info()
    train_dataset = _dataset(data_args, "train", True, data_args.max_train_samples)
    eval_dataset = _dataset(data_args, data_args.eval_split, False, data_args.max_eval_samples)
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
