import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, THIS_DIR)

from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info
from train_ethos_common import EthosModelArguments, run_training


@dataclass
class DataArguments:
    data_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    vocab_dir: str = field(default=".cache/ethos_vocab/ehr_bench")
    task_name: str = field(default="Inpatient_Mortality")
    train_sample_info_path: Optional[str] = field(default=None)
    val_sample_info_path: Optional[str] = field(default=None)
    lazy_mode: bool = field(default=True)
    table_mode: str = field(default="table_only")
    itemid_representation: str = field(default="code")
    concept_map_dir: Optional[str] = field(default=None)
    max_seq_length: int = field(default=4096)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)


def _path(data_args, split):
    custom = data_args.train_sample_info_path if split == "train" else data_args.val_sample_info_path
    return custom or os.path.join(data_args.data_dir, "task_index", split, f"{data_args.task_name}.csv")


def _dataset(data_args, split, shuffle, max_samples):
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    return MIMICIV(
        root_dir=data_args.data_dir,
        sample_info_path=_path(data_args, split),
        lazy_mode=data_args.lazy_mode,
        shuffle=shuffle,
        table_mode=data_args.table_mode,
        max_samples=max_samples,
        itemid_representation=data_args.itemid_representation,
        concept_map_dir=data_args.concept_map_dir,
        return_meds=True,
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
