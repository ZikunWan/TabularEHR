import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import HfArgumentParser, set_seed, Trainer, TrainingArguments

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
ehrshot_dataset_root = os.path.join(project_root, "dataset", "ehrshot")
if ehrshot_dataset_root not in sys.path:
    sys.path.append(ehrshot_dataset_root)

from ehrshot_dataset import EHRSHOTDataset
from models.encoder_classifier import LongTableEncoderClassifier
from models.TableEncoder.config import TableEncoderConfig

from utils.collate import create_collate_fn
from utils.load_embedding import load_embedding_cache, get_embedding
from utils.weight_loader import load_model_weights

@dataclass
class ModelArguments:
    attention_mode: str = field(default='1d', metadata={"help": "Attention mode: '1d', '2d_grid', or 'hierarchical'"})
    use_lora: bool = field(default=False, metadata={"help": "Set True if the checkpoint was saved with LoRA (PEFT) and adapter_config.json is absent/needs override"})
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to base transformer weights (e.g., google/tapas-base) if the model requires them before loading the classifier head/adapter."})

@dataclass
class DataArguments:
    data_dir: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT", metadata={"help": "Root directory for EHRSHOT data"})
    split_info_path: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT/index/ehrshot_test.csv", metadata={"help": "Path to test split csv"})
    embedding_cache: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/.cache/embeddings/ehrshot/text_embeddings.pt", 
                                  metadata={"help": "Path to pre-computed embedding cache"})
    checkpoint_dir: str = field(default=None, metadata={"help": "Path to the checkpoint directory"})
    batch_size: int = field(default=64, metadata={"help": "Evaluation batch size"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit evaluation samples"})
    task_name: str = field(default="lab_anemia", metadata={"help": "The specific task name to test"})
    type_vocab_file: str = field(default="/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/data/type_vocab.json", metadata={"help": "Path to type vocabulary JSON file"})
    seed: int = field(default=42, metadata={"help": "Random seed"})

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

    # 1. Load Embedding Cache
    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    
    # Load Type Vocab defaults
    type_vocab = None
    with open(data_args.type_vocab_file, 'r') as f:
        type_vocab = json.load(f)

    default_config = TableEncoderConfig()
    model_text_dim = default_config.text_dim if text_dim is None else text_dim

    if not os.path.exists(data_args.split_info_path):
        raise FileNotFoundError(f"Test split file not found: {data_args.split_info_path}")

    test_dataset = EHRSHOTDataset(
        root_dir=data_args.data_dir,
        sample_info_path=data_args.split_info_path,
        task_name=data_args.task_name,
        table_mode="table_only",
        max_samples=data_args.max_eval_samples,
    )
    print(f"Test Dataset Size for {data_args.task_name}: {len(test_dataset)}")
    
    if len(test_dataset) == 0:
        print("Dataset is empty. Exiting.")
        sys.exit(0)

    num_classes = 4 if data_args.task_name.startswith("lab_") else 1

    encoder_config = TableEncoderConfig(
        text_dim=model_text_dim,
        attention_mode=model_args.attention_mode,
        type_vocab_size=len(type_vocab),
        num_classes=num_classes,
        problem_type="single_label_classification"
    )

    model = LongTableEncoderClassifier(config=encoder_config)

    # 4. Load weights — auto-detect LoRA vs plain checkpoint
    if model_args.pretrained_path:
        model = load_model_weights(model, model_args.pretrained_path, use_lora=False, is_trainable=False)
        
    if model_args.use_lora:
        from peft import PeftModel
        print(f"Loading LoRA adapter weights from checkpoint: {data_args.checkpoint_dir}")
        model = PeftModel.from_pretrained(model, data_args.checkpoint_dir, is_trainable=False)
    else:
        model = load_model_weights(model, data_args.checkpoint_dir, use_lora=False, is_trainable=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(data_args.checkpoint_dir, "eval_logs"),
        per_device_eval_batch_size=data_args.batch_size,
        remove_unused_columns=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=create_collate_fn(type_vocab),
    )
    
    print("Starting evaluation...")
    predict_outputs = trainer.predict(test_dataset)
    logits = predict_outputs.predictions
    labels_np = predict_outputs.label_ids
    
    results = []
    
    all_targets = labels_np.tolist()
    all_probs = []
    all_preds = []

    if logits.shape[-1] == 1:
        # Binary classification (ensure proper dimensionality)
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
    else:
        # Multi-class classification
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        preds = np.argmax(probs, axis=-1)
        
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

        for i in range(len(labels_np)):
            res = {
                'label': int(labels_np[i]),
                'prob': probs[i].tolist(), # Save class probabilities
                'pred': int(preds[i]),
                'task': data_args.task_name,
            }
            results.append(res)

    # 6. Metrics
    print(f"\n=== Evaluation Results for {data_args.task_name} ===")
    
    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        print("No results collected.")
        return

    # Calculate overall metrics
    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    if num_classes == 1:
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auroc = 0.5
    else:
        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            auroc = 0.5

    acc = accuracy_score(y_true, y_pred)
    
    print(f"{'Task':<20} | {'AUROC':<8} | {'Accuracy':<8} | {'N':<5}")
    print("-" * 55)
    print(f"{data_args.task_name:<20} | {auroc:.4f}   | {acc:.4f}   | {len(y_true):<5}")
    
    final_output = [{
        'task': data_args.task_name,
        'auroc': auroc,
        'accuracy': acc,
        'n_samples': len(y_true)
    }]
    
    # Save results
    output_file = os.path.join(data_args.checkpoint_dir, f"test_results_metrics.csv")
    pd.DataFrame(final_output).to_csv(output_file, index=False)
    print(f"\nMetrics saved to {output_file}")
    
    raw_file = os.path.join(data_args.checkpoint_dir, f"test_raw_predictions.csv")
    df_results.to_csv(raw_file, index=False)
    print(f"Raw predictions saved to {raw_file}")


if __name__ == "__main__":
    main()
