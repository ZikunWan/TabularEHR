import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.renji.renji_dataset import RenjiDataset
from dataset.renji.task_info import get_task_info as get_renji_task_info
from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.ehrshot.task_info import get_task_info as get_ehrshot_task_info
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.eicu.task_info import get_task_info as get_eicu_task_info
from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info as get_mimic_task_info
from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info as get_mimic_iv_cdm_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.query_candidate_decoder import TaskQueryCandidateDecoderModel
from utils.candidate_tasks import build_candidate_embedding_texts, candidate_embedding_keys, get_candidate_texts
from utils.collate import create_candidate_collate_fn, create_single_query_candidate_collate_fn
from utils.load_embedding import (
    build_embedding_matrix,
    build_task_query_embeddings,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.weight_loader import load_encoder_weights, load_task_model_weights


RENJI_ACTIVE_POINTS = ["day30", "day180", "day365"]


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_name: str = field(default="eicu")
    data_dir: str = field(default="")
    processed_dir: Optional[str] = field(default=None)
    sample_info_path: Optional[str] = field(default=None)
    sample_info_test_path: Optional[str] = field(default=None)
    split_info_path: Optional[str] = field(default=None)
    checkpoint_dir: str = field(default="")
    task_name: str = field(default="")
    embedding_cache: str = field(default="")
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_candidate/task_candidate_embeddings.pt")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=512)
    max_table_len: Optional[int] = field(default=None)
    batch_size: int = field(default=64)
    max_eval_samples: Optional[int] = field(default=None)
    seed: int = field(default=42)
    split: str = field(default="test")


def get_dataset_task_info(dataset_name: str):
    if dataset_name == "eicu":
        return get_eicu_task_info()
    if dataset_name == "ehrshot":
        return get_ehrshot_task_info()
    if dataset_name == "mimic_iv_cdm":
        return get_mimic_iv_cdm_task_info()
    if dataset_name == "ehr_bench":
        return get_mimic_task_info()
    if dataset_name == "renji":
        return get_renji_task_info()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def renji_candidate_query(point_key: str, metric: str) -> str:
    _, label_prefix, readable_point = RenjiDataset.PREDICTION_POINTS[point_key]
    return (
        f"Task: predict future {metric} abnormality during {label_prefix} "
        f"after liver transplantation using clinical history up to "
        f"{readable_point} post-transplant."
    )


def build_renji_embedding_texts() -> dict[str, str]:
    texts = {"no": "no", "yes": "yes"}
    for point_key in RENJI_ACTIVE_POINTS:
        for metric in RenjiDataset.ALL_METRICS:
            query = renji_candidate_query(point_key, metric)
            texts[query] = query
    return texts


def resolve_eval_path(data_args: DataArguments):
    return data_args.sample_info_path or data_args.sample_info_test_path or data_args.split_info_path


def build_eval_dataset(data_args: DataArguments):
    eval_path = resolve_eval_path(data_args)
    if data_args.dataset_name == "eicu":
        return EICUDataset(
            root_dir=data_args.data_dir,
            processed_dir=data_args.processed_dir,
            sample_info_path=eval_path,
            task_name=data_args.task_name,
            lazy_mode=True,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
        )
    if data_args.dataset_name == "ehrshot":
        return EHRSHOTDataset(
            root_dir=data_args.data_dir,
            sample_info_path=eval_path,
            task_name=data_args.task_name,
            max_samples=data_args.max_eval_samples,
        )
    if data_args.dataset_name == "mimic_iv_cdm":
        return MIMICIVCDM(
            root_dir=data_args.data_dir,
            split="test",
            task_name=data_args.task_name,
            lazy_mode=False,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
        )
    if data_args.dataset_name == "ehr_bench":
        return MIMICIV(
            root_dir=data_args.data_dir,
            sample_info_path=eval_path,
            lazy_mode=True,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
            use_table_length_cache=False,
        )
    if data_args.dataset_name == "renji":
        return RenjiDataset(
            root_dir=data_args.data_dir,
            split=data_args.split,
            shuffle=False,
            max_samples=data_args.max_eval_samples,
            target_prediction_points=RENJI_ACTIVE_POINTS,
        )
    raise ValueError(f"Unsupported dataset_name: {data_args.dataset_name}")


def label_map_for_task(dataset_name: str, candidate_texts: list[str]):
    return {candidate: idx for idx, candidate in enumerate(candidate_texts)}


def move_tensors_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_metrics(labels, probs):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    preds = probs.argmax(axis=-1).astype(int)
    if probs.shape[1] == 2:
        return {
            "auroc": roc_auc_score(labels, probs[:, 1]) if np.unique(labels).size == 2 else 0.5,
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
        }
    return {
        "auroc": roc_auc_score(labels, probs, multi_class="ovr") if np.unique(labels).size > 1 else 0.5,
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    set_seed(data_args.seed)

    is_multi_query_dataset = data_args.dataset_name == "renji"
    if is_multi_query_dataset:
        task_info = get_dataset_task_info(data_args.dataset_name)[data_args.task_name or "candidate_metric_prediction"]
        candidate_texts = ["no", "yes"]
        query_key = f"{data_args.dataset_name}:{data_args.task_name or 'candidate_metric_prediction'}"
    else:
        task_info = get_dataset_task_info(data_args.dataset_name)[data_args.task_name]
        candidate_texts = get_candidate_texts(task_info)
        query_key = f"{data_args.dataset_name}:{data_args.task_name}"

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]

    with open(data_args.type_vocab_file, "r") as f:
        type_vocab = json.load(f)

    embeddings_by_text, query_dim = build_task_query_embeddings(
        query_texts=build_renji_embedding_texts()
        if is_multi_query_dataset
        else build_candidate_embedding_texts(query_key, task_info["instruction"], candidate_texts),
        cache_path=data_args.query_embedding_cache,
        max_length=data_args.query_max_length,
        knowledge_encoder_path=data_args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
    )
    if is_multi_query_dataset:
        candidate_embeddings_by_text = {
            candidate: embeddings_by_text[candidate]
            for candidate in candidate_texts
        }
        candidate_embeds = None
    else:
        candidate_embeds = torch.stack(
            [embeddings_by_text[key] for key in candidate_embedding_keys(query_key, candidate_texts)]
        ).float()

    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        num_classes=len(candidate_texts),
        problem_type="single_label_classification",
    )
    model = TaskQueryCandidateDecoderModel(config=config, embedding_matrix=embedding_matrix, query_dim=query_dim)
    if model_args.pretrained_path:
        model = load_encoder_weights(model, model_args.pretrained_path)
    model = load_task_model_weights(model, data_args.checkpoint_dir)

    dataset = build_eval_dataset(data_args)
    if is_multi_query_dataset:
        collator = create_candidate_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=embeddings_by_text,
            candidate_embeddings_by_text=candidate_embeddings_by_text,
            include_metadata=True,
        )
    else:
        collator = create_single_query_candidate_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embed=embeddings_by_text[query_key],
            candidate_embeds=candidate_embeds,
            label_map=label_map_for_task(data_args.dataset_name, candidate_texts),
        )
    dataloader = DataLoader(
        dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_probs = []
    all_metadata = []
    with torch.no_grad():
        for batch in dataloader:
            metadata = batch.pop("metadata", None)
            labels = batch["labels"].cpu().numpy()
            batch = move_tensors_to_device(batch, device)
            outputs = model(**batch)
            probs = torch.softmax(outputs.scores.float(), dim=-1).cpu().numpy()
            if is_multi_query_dataset:
                for batch_idx, sample_metadata in enumerate(metadata):
                    for query_idx, item in enumerate(sample_metadata):
                        if labels[batch_idx, query_idx] == -100:
                            continue
                        all_labels.append(int(labels[batch_idx, query_idx]))
                        all_probs.append(probs[batch_idx, query_idx].tolist())
                        all_metadata.append(item)
            else:
                all_labels.extend(labels.tolist())
                all_probs.extend(probs.tolist())

    metrics = compute_metrics(all_labels, all_probs)
    print(f"\n=== Candidate Decoder Evaluation: {data_args.dataset_name}/{data_args.task_name} ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    output_task_name = data_args.task_name or data_args.dataset_name
    output_file = os.path.join(data_args.checkpoint_dir, f"test_results_{output_task_name}.csv")
    rows = []
    probs = np.asarray(all_probs)
    preds = probs.argmax(axis=-1)
    for idx, label in enumerate(all_labels):
        row = {"label": int(label), "prediction": int(preds[idx])}
        if all_metadata:
            row.update(all_metadata[idx])
        for candidate_idx, candidate in enumerate(candidate_texts):
            row[f"prob_{candidate}"] = float(probs[idx, candidate_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"Raw predictions saved to {output_file}")


if __name__ == "__main__":
    main()
