import os
import sys
import pandas as pd
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from transformers import HfArgumentParser, set_seed, Trainer, TrainingArguments

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from dataset.mimic.mimic_dataset import MIMICIV
from dataset.mimic.task_info import get_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.query_classifier import TaskQueryClassificationModel

from utils.collate import create_query_collate_fn
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.load_embedding import build_task_query_embeddings
from utils.weight_loader import load_encoder_weights, load_task_model_weights

@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to base transformer weights."})

@dataclass
class DataArguments:
    data_dir: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular",
        metadata={"help": "Root directory for MIMIC-IV tabular data"}
    )
    task_name: str = field(
        default="ED_Hospitalization",
        metadata={"help": "Task to evaluate (e.g. ED_Hospitalization)"}
    )
    sample_info_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to sample info CSV. If None, uses <data_dir>/task_index/test/<task_name>.csv"}
    )
    embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt",
        metadata={"help": "Path to pre-computed embedding cache"}
    )
    checkpoint_dir: str = field(default=None, metadata={"help": "Path to the checkpoint directory"})
    batch_size: int = field(default=64, metadata={"help": "Evaluation batch size"})
    max_table_len: Optional[int] = field(default=None, metadata={"help": "Keep only the most recent N table rows before encoding"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit evaluation samples"})
    type_vocab_file: str = field(
        default="data/type_vocab.json",
        metadata={"help": "Path to type vocabulary JSON file"}
    )
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt")
    query_encoder: str = field(default="llm")
    query_llm_model_path: str = field(default="/data/model_weights_public/BlueZeros/EHR-R1-1.7B")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=512)
    seed: int = field(default=42, metadata={"help": "Random seed"})
    lazy_mode: bool = field(default=True, metadata={"help": "Load samples lazily from parquet to save memory"})

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    set_seed(data_args.seed)
    
    if not data_args.checkpoint_dir:
        print("Error: --checkpoint_dir must be provided.")
        sys.exit(1)
    if not os.path.isdir(data_args.checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {data_args.checkpoint_dir}")
        sys.exit(1)

    print(f"Checkpoint directory: {data_args.checkpoint_dir}")
    print(f"Task: {data_args.task_name}")

    # 1. Sample info path
    sample_info_path = data_args.sample_info_path
    if sample_info_path is None:
        sample_info_path = os.path.join(data_args.data_dir, "task_index", "test", f"{data_args.task_name}.csv")
    
    if not os.path.exists(sample_info_path):
        raise FileNotFoundError(f"sample_info_path not found: {sample_info_path}")
    
    print(f"Loading task '{data_args.task_name}' from {sample_info_path}...")
    
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
    test_dataset = MIMICIV(
        root_dir=data_args.data_dir,
        sample_info_path=sample_info_path,
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        max_samples=data_args.max_eval_samples,
    )
    print(f"Test dataset size: {len(test_dataset)}")

    task_info = get_task_info()[data_args.task_name]
    query_key = f"ehr_bench:{data_args.task_name}"
    query_embeddings, query_dim = build_task_query_embeddings(
        query_texts={query_key: task_info["instruction"]},
        cache_path=data_args.query_embedding_cache,
        query_encoder=data_args.query_encoder,
        max_length=data_args.query_max_length,
        query_llm_model_path=data_args.query_llm_model_path,
        knowledge_encoder_path=data_args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
    )
    print(f"Query encoder={data_args.query_encoder}, query_dim={query_dim}")

    # 5. Model Config — binary classification
    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        num_classes=1,
        problem_type="single_label_classification"
    )

    model = TaskQueryClassificationModel(
        config=encoder_config,
        embedding_matrix=embedding_matrix,
        query_dim=query_dim,
    )

    # 6. Load weights
    if model_args.pretrained_path:
        model = load_encoder_weights(model, model_args.pretrained_path)
    model = load_task_model_weights(model, data_args.checkpoint_dir)

    training_args = TrainingArguments(
        output_dir=os.path.join(data_args.checkpoint_dir, "eval_logs"),
        per_device_eval_batch_size=data_args.batch_size,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=4,
    )
    
    # 7. Collate function mapped identical to training script
    collate_fn = create_query_collate_fn(
        type_vocab,
        max_table_len=data_args.max_table_len,
        text_to_idx=text_to_idx,
        pad_idx=pad_idx,
        query_embed=query_embeddings[query_key],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
    )
    
    print("Starting evaluation...")
    predict_outputs = trainer.predict(test_dataset)
    logits = predict_outputs.predictions
    labels_np = predict_outputs.label_ids
    
    results = []
    
    all_targets = labels_np.tolist()
    all_probs = []
    all_preds = []

    # Binary classification logic extraction
    probs = 1.0 / (1.0 + np.exp(-logits.squeeze(-1))) # Sigmoid
    preds = (probs > 0.5).astype(int)
    
    all_probs.extend(probs.tolist())
    all_preds.extend(preds.tolist())
    
    for i in range(len(labels_np)):
        res = {
            'label': int(labels_np[i]),
            'prob': float(probs[i]),
            'pred': int(preds[i]),
            'task': data_args.task_name,
        }
        results.append(res)

    # 8. Metrics
    print(f"\n=== Evaluation Results for {data_args.task_name} ===")
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("No results collected.")
        return

    # Calculate overall metrics
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5

    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = 0.0

    acc = accuracy_score(y_true, y_pred)
    
    print(f"{'Task':<22} | {'AUROC':<8} | {'AUPRC':<8} | {'Accuracy':<8} | {'N':<5}")
    print("-" * 65)
    print(f"{data_args.task_name:<22} | {auroc:.4f}   | {auprc:.4f}   | {acc:.4f}   | {len(y_true):<5}")
    
    final_output = [{
        'task': data_args.task_name,
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'n_samples': len(y_true)
    }]
    
    # Save results
    output_file = os.path.join(data_args.checkpoint_dir, "test_results_metrics.csv")
    pd.DataFrame(final_output).to_csv(output_file, index=False)
    print(f"\nMetrics saved to {output_file}")
    
    raw_file = os.path.join(data_args.checkpoint_dir, "test_raw_predictions.csv")
    df_results.to_csv(raw_file, index=False)
    print(f"Raw predictions saved to {raw_file}")


if __name__ == "__main__":
    main()
