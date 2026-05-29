import os
import sys
import json
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed, EarlyStoppingCallback
from transformers.utils import logging as hf_logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in [-1, 0]:
        print(*args, **kwargs)


def quiet_non_main_process_logs():
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank not in [-1, 0]:
        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)
        logging.getLogger("deepspeed").setLevel(logging.ERROR)

from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.query_classifier import TaskQueryClassificationModel
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.metrics import compute_classification_metrics
from utils.samplers import TrainerWithBatchSampler, build_train_batch_sampler
from utils.weight_loader import load_model_weights
from utils.collate import create_query_collate_fn
from utils.query_embedding import build_query_embeddings

# All EHR-Bench risk prediction tasks
ALL_RISK_PREDICTION_TASKS = [
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

LABEL_MAP = {"yes": 1, "no": 0}

@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to pre-trained model checkpoint"})
    use_lora: bool = field(default=False, metadata={"help": "Apply LoRA adapters to encoder"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="qkv,proj,w12,w3",
        metadata={"help": "Comma-separated list of Linear layer names to apply LoRA to."}
    )

@dataclass
class DataArguments:
    max_table_len: int = field(metadata={"help": "Keep only the most recent N table rows before encoding"})
    data_dir: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular",
        metadata={"help": "Root directory for MIMIC-IV tabular data"}
    )
    task_name: str = field(
        default="ED_Hospitalization",
        metadata={"help": f"Task to train. One of: {ALL_RISK_PREDICTION_TASKS}"}
    )
    train_sample_info_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to train sample-info CSV. If None, uses <data_dir>/task_index/train/<task_name>.csv"}
    )
    val_sample_info_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to val sample-info CSV. If None, uses <data_dir>/task_index/val/<task_name>.csv"}
    )
    embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt",
        metadata={"help": "Path to pre-computed embedding cache"}
    )
    type_vocab_file: str = field(
        default="/data/zikun_workspace/code/data/type_vocab.json",
        metadata={"help": "Path to unified type vocabulary JSON file"}
    )
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt")
    query_llm_model_path: str = field(default="/data/model_weights_public/BlueZeros/EHR-R1-1.7B")
    query_max_length: int = field(default=512)
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Limit training samples"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit evaluation samples"})
    lazy_mode: bool = field(default=True, metadata={"help": "Load samples lazily from parquet to save memory"})
    max_tokens_per_batch: Optional[int] = field(default=None, metadata={"help": "Enable ApproxBatchSampler when >0. This caps padded tokens per batch."})
    use_sortish_sampler: bool = field(default=True, metadata={"help": "Whether to use SortishSampler before ApproxBatchSampler packing."})
    sortish_chunk_factor: int = field(default=50, metadata={"help": "Sortish chunk factor. Larger means more sorting, less randomness."})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="EHR-Bench-Classifier", metadata={"help": "W&B project name."})
    early_stopping_patience: int = field(default=5, metadata={"help": "Number of eval steps with no improvement before stopping. Set to 0 to disable."})

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    quiet_non_main_process_logs()

    # Validate task
    if data_args.task_name not in ALL_RISK_PREDICTION_TASKS:
        raise ValueError(f"task_name '{data_args.task_name}' not in supported risk prediction tasks: {ALL_RISK_PREDICTION_TASKS}")

    # Default training settings
    training_args.lr_scheduler_type = "cosine"
    training_args.warmup_steps = 10
    training_args.warmup_ratio = 0.0
    training_args.weight_decay = 0.01
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 10
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 10
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.save_total_limit = 1
    training_args.bf16 = True
    training_args.dataloader_num_workers = 32
    training_args.remove_unused_columns = False
    training_args.report_to = ["wandb"]
    training_args.save_safetensors = True
    
    # Enforce best model logic
    if training_args.eval_strategy != "no":
        training_args.metric_for_best_model = "auroc"
        training_args.greater_is_better = True
        training_args.load_best_model_at_end = True

    set_seed(training_args.seed)

    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # 1. Train/Val sample info paths (separate files)
    train_sample_info_path = data_args.train_sample_info_path
    if train_sample_info_path is None:
        train_sample_info_path = os.path.join(data_args.data_dir, "task_index", "train", f"{data_args.task_name}.csv")

    val_sample_info_path = data_args.val_sample_info_path
    if val_sample_info_path is None:
        val_sample_info_path = os.path.join(data_args.data_dir, "task_index", "val", f"{data_args.task_name}.csv")

    if not os.path.exists(train_sample_info_path):
        raise FileNotFoundError(f"train_sample_info_path not found: {train_sample_info_path}")
    if not os.path.exists(val_sample_info_path):
        raise FileNotFoundError(f"val_sample_info_path not found: {val_sample_info_path}")

    rank0_print(f"Loading TRAIN task '{data_args.task_name}' from {train_sample_info_path}...")
    rank0_print(f"Loading VAL task '{data_args.task_name}' from {val_sample_info_path}...")

    # 2. Load Embedding Cache
    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]

    # 3. Load Type Vocab
    with open(data_args.type_vocab_file, 'r') as f:
        type_vocab = json.load(f)

    # 4. Dataset
    train_dataset = MIMICIV(
        root_dir=data_args.data_dir,
        sample_info_path=train_sample_info_path,
        lazy_mode=data_args.lazy_mode,
        table_mode="table_only",
        shuffle=True,
        max_samples=data_args.max_train_samples,
    )
    
    val_dataset = MIMICIV(
        root_dir=data_args.data_dir,
        sample_info_path=val_sample_info_path,
        lazy_mode=data_args.lazy_mode,
        table_mode="table_only",
        shuffle=False,
        max_samples=data_args.max_eval_samples,
    )
    
    rank0_print(f"Train dataset size: {len(train_dataset)}")
    rank0_print(f"Val dataset size: {len(val_dataset)}")

    task_info = get_task_info()[data_args.task_name]
    query_key = f"ehr_bench:{data_args.task_name}"
    query_embeddings, query_dim = build_query_embeddings(
        {query_key: task_info["instruction"]},
        data_args.query_embedding_cache,
        data_args.query_llm_model_path,
        data_args.query_max_length,
    )

    # 5. Model Config — binary classification with BCEWithLogitsLoss
    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        num_classes=1,                       # Single binary output
        problem_type="single_label_classification"  # triggers BCEWithLogitsLoss via num_classes=1
    )

    model = TaskQueryClassificationModel(
        config=encoder_config,
        embedding_matrix=embedding_matrix,
        query_dim=query_dim,
    )

    # 6. Load pre-trained weights
    if model_args.pretrained_path:
        rank0_print(f"Loading pretrained weights from {model_args.pretrained_path}")
        model = load_model_weights(model, model_args.pretrained_path, use_lora=False, is_trainable=False)

    # 7. LoRA (optional)
    if model_args.use_lora:
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(',')]
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=["classifier", "query_head"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        rank0_print("LoRA applied successfully.")

    # 8. Collate function
    collate_fn = create_query_collate_fn(
        type_vocab,
        label_map=LABEL_MAP,
        max_table_len=data_args.max_table_len,
        text_to_idx=text_to_idx,
        pad_idx=pad_idx,
        query_embed=query_embeddings[query_key],
    )

    # 9. Trainer
    callbacks = []
    if getattr(training_args, "early_stopping_patience", 0) > 0 and training_args.eval_strategy != "no":
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
        rank0_print(f"Early stopping enabled with patience={training_args.early_stopping_patience}")

    base_run_name = training_args.run_name
    training_args.run_name = f"{base_run_name}__{data_args.task_name}"

    trainer_cls = Trainer
    trainer_kwargs = {}
    if data_args.max_tokens_per_batch is not None and data_args.max_tokens_per_batch > 0:
        train_batch_sampler = build_train_batch_sampler(
            dataset=train_dataset,
            per_device_batch_size=training_args.per_device_train_batch_size,
            max_tokens_per_batch=data_args.max_tokens_per_batch,
            use_sortish_sampler=data_args.use_sortish_sampler,
            sortish_chunk_factor=data_args.sortish_chunk_factor,
            shuffle=True,
            seed=training_args.seed,
            drop_last=training_args.dataloader_drop_last,
        )
        trainer_cls = TrainerWithBatchSampler
        trainer_kwargs["train_batch_sampler"] = train_batch_sampler
        rank0_print(
            f"Enabled dynamic token batching: max_tokens_per_batch={data_args.max_tokens_per_batch}, "
            f"use_sortish_sampler={data_args.use_sortish_sampler}"
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_classification_metrics,
        callbacks=callbacks if callbacks else None,
        **trainer_kwargs,
    )

    rank0_print(f"\nStarting training for task: {data_args.task_name}")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    rank0_print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()


if __name__ == "__main__":
    main()
