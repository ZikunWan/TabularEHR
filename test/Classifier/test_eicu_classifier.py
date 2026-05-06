import os
import sys
import pandas as pd
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import HfArgumentParser, set_seed, Trainer, TrainingArguments

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.eicu.eicu_dataset import EICUDataset
from models.encoder_classifier import LongTableEncoderClassifier
from models.TableEncoder.config import LongTableEncoderMemoryConfig

from utils.collate import create_collate_fn
from utils.load_embedding import load_embedding_cache
from utils.weight_loader import load_model_weights

@dataclass
class ModelArguments:
    use_lora: bool = field(default=False, metadata={"help": "Set True if the checkpoint was saved with LoRA (PEFT)"})
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to base transformer weights (e.g., google/tapas-base)."})

@dataclass
class DataArguments:
    data_dir: str = field(default="/home/ma-user/sfs_turbo/Data/eicu-crd/2.0", metadata={"help": "Root directory for eICU data"})
    processed_dir: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/eicu-crd/processed", metadata={"help": "Path to processed eICU folder"})
    sample_info_val_path: Optional[str] = field(default=None, metadata={"help": "Path to eICU val sample-info JSON"})
    sample_info_test_path: Optional[str] = field(default=None, metadata={"help": "Path to eICU test sample-info JSON"})
    embedding_cache: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/.cache/embeddings/eicu/text_embeddings.pt", metadata={"help": "Path to pre-computed embedding cache"})
    checkpoint_dir: str = field(default=None, metadata={"help": "Path to the checkpoint directory"})
    batch_size: int = field(default=64, metadata={"help": "Evaluation batch size"})
    max_table_len: Optional[int] = field(default=None, metadata={"help": "Keep only the most recent N table rows before encoding"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit evaluation samples"})
    task_name: str = field(default="mortality", metadata={"help": "The specific task name to test"})
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
    _, text_dim = load_embedding_cache(data_args.embedding_cache)

    # Load Type Vocab defaults
    type_vocab = None
    with open(data_args.type_vocab_file, 'r') as f:
        type_vocab = json.load(f)

    # 2. Load val + test split and merge them
    print(f"Loading eICU dataset from {data_args.data_dir}...")
    sample_info_val_path = data_args.sample_info_val_path
    if sample_info_val_path is None:
        sample_info_val_path = os.path.join(data_args.processed_dir, "sample_info_val.json")
    sample_info_test_path = data_args.sample_info_test_path
    if sample_info_test_path is None:
        sample_info_test_path = os.path.join(data_args.processed_dir, "sample_info_test.json")
    if not os.path.exists(sample_info_val_path):
        raise FileNotFoundError(f"Val sample info not found: {sample_info_val_path}")
    if not os.path.exists(sample_info_test_path):
        raise FileNotFoundError(f"Test sample info not found: {sample_info_test_path}")

    val_dataset = EICUDataset(
        root_dir=data_args.data_dir,
        processed_dir=data_args.processed_dir,
        sample_info_path=sample_info_val_path,
        task_name=data_args.task_name,
        table_mode="table_only",
        lazy_mode=False,
        shuffle=False,
    )
    test_dataset = EICUDataset(
        root_dir=data_args.data_dir,
        processed_dir=data_args.processed_dir,
        sample_info_path=sample_info_test_path,
        task_name=data_args.task_name,
        table_mode="table_only",
        lazy_mode=False,
        shuffle=False,
    )
    
    # Merge val into test
    test_dataset.sample_info = val_dataset.sample_info + test_dataset.sample_info
    test_dataset.data = val_dataset.data + test_dataset.data
    print(f"Merged val ({len(val_dataset)}) + test ({len(test_dataset) - len(val_dataset)}) = {len(test_dataset)} samples")
    
    if data_args.max_eval_samples:
        test_dataset.sample_info = test_dataset.sample_info[:data_args.max_eval_samples]
        test_dataset.data = test_dataset.data[:data_args.max_eval_samples]
        print(f"Truncated to {len(test_dataset)} samples.")

    if len(test_dataset) == 0:
        print("Dataset is empty. Exiting.")
        sys.exit(0)

    # Determine num_classes and problem_type dynamically from dataset info
    first_sample = test_dataset[0]
    task_type = first_sample['task_info']['task_type']
    
    num_classes = 1
    problem_type = "single_label_classification"
    label_map = None
    
    if task_type == "binary_classification":
        num_classes = 1
        problem_type = "single_label_classification"
        label_map = None
    elif task_type == "multi_label_classification":
        num_classes = int(first_sample["task_info"].get("num_classes", 28))
        problem_type = "multi_label_classification"
    else:
        num_classes = int(first_sample["task_info"].get("num_classes", 0))
        if num_classes == 0:
            if data_args.task_name == 'wbc': num_classes = 3
            elif data_args.task_name in ['creatinine', 'bilirubin', 'platelets']: num_classes = 5
            else: num_classes = 2
        problem_type = "single_label_classification"
        
    print(f"Task type: {task_type}")
    print(f"Num classes: {num_classes}, Problem type: {problem_type}")

    encoder_config = LongTableEncoderMemoryConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        num_classes=num_classes,
        problem_type=problem_type
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
        data_collator=create_collate_fn(type_vocab, label_map=label_map, max_table_len=data_args.max_table_len),
    )
    
    print("Starting evaluation...")
    predict_outputs = trainer.predict(test_dataset)
    logits = predict_outputs.predictions
    labels_np = predict_outputs.label_ids
    
    y_true = np.array(labels_np)
    
    all_probs = []
    all_preds = []

    if problem_type == "multi_label_classification":
        # Multi-label classification (Diagnosis)
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        preds = (probs > 0.5).astype(int)
        
        # Micro AUROC across all classes
        try:
            auroc = roc_auc_score(y_true, probs, average='micro')
        except ValueError:
            auroc = 0.5
            
        acc = accuracy_score(y_true, preds)
        
    elif num_classes == 1:
        # Binary Risk Prediction
        probs = 1 / (1 + np.exp(-np.array(logits).squeeze(-1)))
        preds = (probs > 0.5).astype(int)
        
        try:
            auroc = roc_auc_score(y_true, probs)
        except ValueError:
            auroc = 0.5
            
        acc = accuracy_score(y_true, preds)
        
    else:
        # Multi-class Target (Lab value grading)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        preds = np.argmax(probs, axis=-1)
        
        try:
            auroc = roc_auc_score(y_true, probs, multi_class='ovr')
        except ValueError:
            auroc = 0.5
            
        acc = accuracy_score(y_true, preds)

    # 6. Metrics
    print(f"\n=== Evaluation Results for {data_args.task_name} ===")
    print(f"{'Task':<20} | {'AUROC':<8} | {'Accuracy':<8} | {'N':<5}")
    print("-" * 55)
    print(f"{data_args.task_name[:20]:<20} | {auroc:.4f}   | {acc:.4f}   | {len(y_true):<5}")
    
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


if __name__ == "__main__":
    main()
