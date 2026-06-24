import os
import sys
import json
import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import glob
from sklearn.metrics import roc_auc_score
from torch.utils.data import Subset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, HfArgumentParser, set_seed
from transformers.utils import logging as hf_logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def rank0_print(*args, **kwargs):
    rank = os.environ.get("RANK")
    if rank is not None:
        if int(rank) == 0:
            print(*args, **kwargs)
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in [-1, 0]:
        print(*args, **kwargs)


def quiet_non_main_process_logs():
    rank = os.environ.get("RANK")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_non_main = int(rank) != 0 if rank is not None else local_rank not in [-1, 0]
    if is_non_main:
        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)
        logging.getLogger("deepspeed").setLevel(logging.ERROR)

from dataset.renji.renji_dataset import RenjiDataset
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.query_classifier import TaskQueryClassificationModel
from utils.weight_loader import apply_fine_tune_mode, load_encoder_weights
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.collate import create_query_collate_fn
from utils.multilabel_split import select_balanced_multilabel_groups
from utils.load_embedding import build_task_query_embeddings

ACTIVE_POINTS = ["day30", "day180", "day365"]

@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to pre-trained model checkpoint"})
    fine_tune_mode: str = field(
        default="full_fine_tune",
        metadata={"help": "Fine-tuning mode: full_fine_tune or linear_probe"},
    )


@dataclass
class DataArguments:
    max_table_len: int = field(metadata={"help": "Keep only the most recent N table rows before encoding"})
    data_dir: str = field(default="/data/EHR_data_public/Renji")
    embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt")
    max_train_samples: Optional[int] = field(default=None)
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt")
    query_encoder: str = field(default="llm")
    query_llm_model_path: str = field(default="/data/model_weights_public/BlueZeros/EHR-R1-1.7B")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=512)
    monitor_fraction: float = field(
        default=0.0,
        metadata={"help": "Patient-level fraction held out from train for balanced monitoring. Set to 0 to disable."},
    )
    monitor_seed: int = field(default=42)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="Renji")
    lr_scheduler_type: str = field(default="cosine")
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Number of evaluations without AUROC improvement before stopping. Set to 0 to disable."},
    )


def _build_patient_label_vectors(dataset):
    num_labels = len(ACTIVE_POINTS) * len(RenjiDataset.ALL_METRICS)
    vectors = {}
    for sample in dataset.samples:
        patient_id = sample["fname_key"]
        labels = vectors.setdefault(patient_id, [None] * num_labels)
        patient_labels = dataset.labels_df.loc[patient_id]
        point_key = sample["prediction_point"]
        point_idx = ACTIVE_POINTS.index(point_key)
        _, label_prefix, _ = RenjiDataset.PREDICTION_POINTS[point_key]
        for metric_idx, metric in enumerate(RenjiDataset.ALL_METRICS):
            column = f"{label_prefix}_{metric}"
            if column in patient_labels and pd.notna(patient_labels[column]):
                labels[point_idx * len(RenjiDataset.ALL_METRICS) + metric_idx] = int(
                    patient_labels[column]
                )
    return vectors


def _split_train_monitor_dataset(dataset, monitor_fraction, seed):
    if not 0.0 < monitor_fraction < 1.0:
        raise ValueError("monitor_fraction must be between 0 and 1")

    labels_by_patient = _build_patient_label_vectors(dataset)
    monitor_patients_count = max(1, round(len(labels_by_patient) * monitor_fraction))
    monitor_patients = set(
        select_balanced_multilabel_groups(
            labels_by_group=labels_by_patient,
            holdout_size=monitor_patients_count,
            seed=seed,
        )
    )
    train_indices = [
        idx for idx, sample in enumerate(dataset.samples) if sample["fname_key"] not in monitor_patients
    ]
    monitor_indices = [
        idx for idx, sample in enumerate(dataset.samples) if sample["fname_key"] in monitor_patients
    ]

    monitor_vectors = [labels_by_patient[patient_id] for patient_id in monitor_patients]
    balanced_tasks = 0
    absolute_balance_errors = []
    for label_idx in range(len(monitor_vectors[0])):
        values = [vector[label_idx] for vector in monitor_vectors if vector[label_idx] is not None]
        positives = sum(values)
        negatives = len(values) - positives
        if positives > 0 and negatives > 0:
            balanced_tasks += 1
            absolute_balance_errors.append(abs(positives / len(values) - 0.5))

    mean_balance_error = (
        sum(absolute_balance_errors) / len(absolute_balance_errors)
        if absolute_balance_errors
        else float("nan")
    )
    rank0_print(
        f"Balanced monitor split: patients={len(monitor_patients)}/{len(labels_by_patient)}, "
        f"samples={len(monitor_indices)}, train_samples={len(train_indices)}, "
        f"tasks_with_both_classes={balanced_tasks}, "
        f"mean_abs_positive_rate_minus_0.5={mean_balance_error:.4f}"
    )
    return Subset(dataset, train_indices), Subset(dataset, monitor_indices)


def compute_renji_metrics(eval_pred):
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    labels = np.asarray(eval_pred.label_ids)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    mask = labels != -100

    if not np.any(mask):
        return {"auroc": 0.5, "micro_auroc": 0.5, "accuracy": 0.0, "evaluated_tasks": 0}

    flat_labels = labels[mask].astype(int)
    flat_probabilities = probabilities[mask]
    flat_predictions = (flat_probabilities >= 0.5).astype(int)
    micro_auroc = (
        roc_auc_score(flat_labels, flat_probabilities)
        if np.unique(flat_labels).size == 2
        else 0.5
    )

    task_aurocs = []
    for point_idx in range(labels.shape[1]):
        for metric_idx in range(labels.shape[2]):
            task_mask = mask[:, point_idx, metric_idx]
            task_labels = labels[task_mask, point_idx, metric_idx].astype(int)
            if np.unique(task_labels).size != 2:
                continue
            task_probabilities = probabilities[task_mask, point_idx, metric_idx]
            task_aurocs.append(roc_auc_score(task_labels, task_probabilities))

    return {
        "auroc": float(np.mean(task_aurocs)) if task_aurocs else 0.5,
        "micro_auroc": float(micro_auroc),
        "accuracy": float(np.mean(flat_predictions == flat_labels)),
        "positive_rate": float(np.mean(flat_labels)),
        "evaluated_tasks": len(task_aurocs),
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    quiet_non_main_process_logs()
    
    if training_args.warmup_steps == 0 and not training_args.warmup_ratio:
        training_args.warmup_steps = 100
    training_args.weight_decay = 0.01
    training_args.adam_epsilon = 1e-6
    training_args.seed = 42
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
    training_args.logging_nan_inf_filter = False
    
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    training_args.ddp_find_unused_parameters = False
    set_seed(training_args.seed)

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]
    
    vocab_path = os.path.join(project_root, data_args.type_vocab_file)
    with open(vocab_path, 'r') as f:
        type_vocab = json.load(f)

    train_dataset = RenjiDataset(
        root_dir=data_args.data_dir, split="train", shuffle=True,
        max_samples=data_args.max_train_samples,
        target_prediction_points=ACTIVE_POINTS,
    )
    monitor_dataset = None
    if data_args.monitor_fraction > 0:
        if training_args.eval_strategy == "no":
            raise ValueError("monitor_fraction requires --eval_strategy steps or epoch")
        train_dataset, monitor_dataset = _split_train_monitor_dataset(
            dataset=train_dataset,
            monitor_fraction=data_args.monitor_fraction,
            seed=data_args.monitor_seed,
        )
        training_args.metric_for_best_model = "auroc"
        training_args.greater_is_better = True
        training_args.load_best_model_at_end = True

    query_texts = {}
    query_template = RenjiDataset.TASK_INFO["multi_label_prediction"]["instruction_template"]
    for point_key in ACTIVE_POINTS:
        _, _, readable_point = RenjiDataset.TASK_PREDICTION_POINTS[point_key]
        instruction = query_template.format(prediction_point=f"{readable_point} post-transplant")
        query_texts[instruction] = instruction
    query_embeddings_by_text, query_dim = build_task_query_embeddings(
        query_texts=query_texts,
        cache_path=data_args.query_embedding_cache,
        query_encoder=data_args.query_encoder,
        max_length=data_args.query_max_length,
        query_llm_model_path=data_args.query_llm_model_path,
        knowledge_encoder_path=data_args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
    )
    rank0_print(f"Query encoder={data_args.query_encoder}, query_dim={query_dim}")

    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        num_points=len(ACTIVE_POINTS),
        num_metrics=len(RenjiDataset.ALL_METRICS),
        num_classes=len(ACTIVE_POINTS) * len(RenjiDataset.ALL_METRICS),
        problem_type="multi_label_classification"
    )

    model = TaskQueryClassificationModel(
        config=encoder_config,
        embedding_matrix=embedding_matrix,
        query_dim=query_dim,
    )
    model = load_encoder_weights(model, model_args.pretrained_path)
    model = apply_fine_tune_mode(model, model_args.fine_tune_mode, log_fn=rank0_print)

    callbacks = []
    if monitor_dataset is not None and training_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience,
            )
        )
        rank0_print(
            f"Early stopping enabled: patience={training_args.early_stopping_patience} "
            f"evaluations, metric=eval_{training_args.metric_for_best_model}"
        )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset,
        eval_dataset=monitor_dataset,
        data_collator=create_query_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=query_embeddings_by_text,
        ),
        compute_metrics=compute_renji_metrics if monitor_dataset is not None else None,
        callbacks=callbacks if callbacks else None,
    )

    resume_ckpt = None
    if training_args.resume_from_checkpoint:
        rfc = training_args.resume_from_checkpoint
        if isinstance(rfc, str) and rfc.lower() not in ("true", "1", "yes"): resume_ckpt = rfc
        else:
            ckpt_dirs = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
            if ckpt_dirs: resume_ckpt = ckpt_dirs[-1]

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model()


if __name__ == "__main__":
    main()
