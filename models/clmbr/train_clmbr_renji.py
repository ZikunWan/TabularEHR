"""
Training script for fine-tuning CLMBR-T-Base on Renji dataset using Hugging Face Trainer.
"""

import argparse
import json
import os
import pickle
import sys
from typing import List, Dict, Any, Optional
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback

# femr imports
import femr.models.transformer
import femr.models.tokenizer
import femr.models.processor

# Add project root to path for other utilities if needed
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)


@dataclass
class ModelArguments:
    model_path: str = "/home/ma-user/sfs_turbo/model_weights/clmbr-t-base"
    dropout: float = 0.1
    dim: int = 768

@dataclass
class DataArguments:
    data_dir: str = "/home/ma-user/sfs_turbo/sai6/zkwan/Renji/meds_data"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None


class RenjiMEDSDataset(Dataset):
    """
    Dataset for Renji data in MEDS format (list of dicts).
    """
    def __init__(self, data_path: str, split: str = "train", max_samples: int = None):
        self.data_path = data_path
        self.split = split
        
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            self.all_patients = pickle.load(f)
        
        # We assume data is already split into files.
        self.indices = np.arange(len(self.all_patients))
            
        if max_samples:
            self.indices = self.indices[:max_samples]
            
        print(f"Loaded {len(self.indices)} patients for split '{split}'")
        
        # Prediction points configuration
        self.PREDICTION_POINTS = {
            'day14': (14, '2w-1m'),
            'day30': (30, '2m-6m'),
            'day180': (180, '7m-12m'),
            'day365': (365, '1y+'),
        }
        
        # Deterministic Mappings from labels.csv headers to internal metrics
        self.CSV_SUFFIX_MAP = {
            'ALB': 'ALB', 
            'ALP': 'ALP',
            'ALT': 'ALT',
            'AST': 'AST',
            'CR': 'CR',
            'DB': 'DB',
            'HB': 'HB',
            'INR': 'INR',
            'N(%)': 'N_Percent',
            'PLT': 'PLT',
            'PT': 'PT',
            'TB': 'TB',
            'TP': 'TP',
            'WBC': 'WBC',
            'γ-GT': 'GGT',
            '他克莫司浓度': 'Tacrolimus_Conc',
            #'嗜酸性粒细胞百分比': 'Eosinophil_Percent',
            '尿酸': 'Uric_Acid',
            '总胆固醇': 'Cholesterol',
            '淋巴细胞绝对值': 'Lymphocyte_Abs',
            '环孢素峰浓度': 'CsA_Peak',
            '环孢素谷浓度': 'CsA_Trough',
            '甘油三脂': 'Triglyceride',
            '胆汁酸': 'Bile_Acid',
            #'血氨': 'Blood_Ammonia',
            '血糖': 'Glucose'
        }
        
        self.ALL_METRICS = sorted(list(set(self.CSV_SUFFIX_MAP.values())))
        
        self.metric_to_idx = {m: i for i, m in enumerate(self.ALL_METRICS)}
        self.point_to_idx = {p: i for i, p in enumerate(self.PREDICTION_POINTS.keys())}
        
        # Pre-compute full key lookup: csv_col -> (point_idx, metric_idx)
        self.label_lookup = {}
        for point_key, (cutoff, prefix) in self.PREDICTION_POINTS.items():
            point_idx = self.point_to_idx[point_key]
            for suffix, metric_name in self.CSV_SUFFIX_MAP.items():
                if metric_name in self.metric_to_idx:
                    metric_idx = self.metric_to_idx[metric_name]
                    csv_col = f"{prefix}_{suffix}"
                    self.label_lookup[csv_col] = (point_idx, metric_idx)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        patient_idx = self.indices[idx]
        patient = self.all_patients[patient_idx]
        
        # Prepare labels
        # labels should be a Tensor of shape (num_points, num_metrics)
        # containing 0, 1, or -100 (ignore)
        
        label_matrix = torch.full((len(self.PREDICTION_POINTS), len(self.ALL_METRICS)), -100, dtype=torch.long)
        
        if 'labels' in patient:
            raw_labels = patient['labels']
            for col, val in raw_labels.items():
                if col in self.label_lookup:
                    point_idx, metric_idx = self.label_lookup[col]
                    
                    # Parse value
                    try:
                        fval = float(val)
                        if not np.isnan(fval):
                            label_matrix[point_idx, metric_idx] = int(fval)
                    except (ValueError, TypeError):
                        pass

        return {
            'patient': patient,
            'labels': label_matrix
        }


class DataCollatorForCLMBR:
    def __init__(self, batch_processor):
        self.batch_processor = batch_processor
        
    def __call__(self, batch):
        patients = [item['patient'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        creator = self.batch_processor.creator
        creator.start_batch()
        for patient in patients:
            creator.add_patient(patient)
            
        batch_data = creator.get_batch_data()
        batch_data = creator.cleanup_batch(batch_data)
        
        # 转换 Tensor 并升维 (保持不变)
        def to_tensor_and_unsqueeze(d):
            if isinstance(d, dict):
                return {k: to_tensor_and_unsqueeze(v) for k, v in d.items()}
            elif isinstance(d, np.ndarray):
                if np.issubdtype(d.dtype, np.integer):
                    t = torch.from_numpy(d).long()
                elif np.issubdtype(d.dtype, np.floating):
                    t = torch.from_numpy(d).float()
                elif d.dtype == bool:
                    t = torch.from_numpy(d).bool()
                else:
                    t = torch.from_numpy(d)
                return t.unsqueeze(0)
            return d

        model_inputs = {"batch": to_tensor_and_unsqueeze(batch_data)}
        labels_tensor = torch.stack(labels)
        
        batch_dict = {k: v for k, v in model_inputs.items()}
        batch_dict['labels'] = labels_tensor
        # 移除 batch_indices
        
        return batch_dict


class CLMBRForRenji(nn.Module):
    # __init__ 保持不变
    def __init__(self, model_path, num_points, num_metrics, dropout=0.1):
        super().__init__()
        self.clmbr = femr.models.transformer.FEMRModel.from_pretrained(model_path)
        self.config = self.clmbr.config
        self.dropout = nn.Dropout(dropout)
        
        hidden_size = self.config.transformer_config.hidden_size
        
        self.classifier = nn.Linear(hidden_size, num_points * num_metrics)
        self.num_points = num_points
        self.num_metrics = num_metrics

    def forward(self, labels=None, **batch_inputs):
        outputs = self.clmbr(**batch_inputs)
        
        if isinstance(outputs, tuple):
            result = outputs[1]
        else:
            result = outputs
            
        representations = result['representations'] 
        
        if 'batch' in batch_inputs:
            patient_lengths = batch_inputs['batch']['transformer']['patient_lengths']
        else:
            patient_lengths = batch_inputs['transformer']['patient_lengths']
            
        # 展平 lengths
        split_sections = patient_lengths.view(-1).cpu().tolist()
        
        # 切分
        per_patient_reps = torch.split(representations, split_sections, dim=0)
        
        # Mean pooling
        pooled_list = []
        for pat_rep in per_patient_reps:
            pooled_list.append(pat_rep.mean(dim=0)) 
            
        pooled_output = torch.stack(pooled_list)
        
        # --- DDP 不需要手动 slice，每张卡处理的数据就是它自己的 ---
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) 
        logits = logits.reshape(-1, self.num_points, self.num_metrics)
        
        loss = None
        if labels is not None:
            # 这里的形状检查建议保留，以防万一
            if labels.shape[0] != logits.shape[0]:
                 min_n = min(labels.shape[0], logits.shape[0])
                 labels = labels[:min_n]
                 logits = logits[:min_n]

            loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss_matrix = loss_fct(logits, labels.float())
            
            mask = (labels != -100)
            masked_loss = loss_matrix * mask.float()
            
            num_valid = mask.sum()
            if num_valid > 0:
                loss = masked_loss.sum() / num_valid
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                
        return (loss, logits) if loss is not None else logits

def compute_metrics(eval_pred: EvalPrediction):
    # eval_pred is a tuple (predictions, labels)
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    
    # Handle tuple output (if model returns (loss, logits))
    if isinstance(logits, tuple):
        logits = logits[0]
        
    # Flatten everything
    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    
    # --- Fix: Ensure shapes match before masking ---
    # In rare DataParallel edge cases, gathered logits might be smaller than labels
    min_len = min(len(flat_logits), len(flat_labels))
    
    if len(flat_logits) != len(flat_labels):
        print(f"Warning: Shape mismatch in compute_metrics. Logits: {len(flat_logits)}, Labels: {len(flat_labels)}")
        # Truncate to the shorter length to avoid crash
        flat_logits = flat_logits[:min_len]
        flat_labels = flat_labels[:min_len]

    # Create mask
    mask = flat_labels != -100
    
    # Apply mask
    valid_logits = flat_logits[mask]
    valid_labels = flat_labels[mask]
    
    metrics = {}
    
    if len(valid_labels) > 0 and len(np.unique(valid_labels)) > 1:
        try:
            # Sigmoid for probabilities
            preds = 1 / (1 + np.exp(-valid_logits))
            auc = roc_auc_score(valid_labels, preds)
            metrics['auroc'] = auc
        except Exception as e:
            print(f"Metrics Error: {e}")
            metrics['auroc'] = 0.0
    else:
        metrics['auroc'] = 0.0
    
    return metrics

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/ma-user/sfs_turbo/model_weights/clmbr-t-base")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data_dir", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/Renji/meds_data")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/clmbr_renji_v1")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    
    args = parser.parse_args()
    
    # 1. Setup Data
    train_path = os.path.join(args.data_dir, "renji_meds_train.pkl")
    test_path = os.path.join(args.data_dir, "renji_meds_test.pkl")
    
    if not os.path.exists(train_path):
        print(f"Error: Train data not found at {train_path}")
        return
        
    train_dataset = RenjiMEDSDataset(train_path, split="train", max_samples=args.max_train_samples)
    
    eval_dataset = None
    if os.path.exists(test_path):
        eval_dataset = RenjiMEDSDataset(test_path, split="test", max_samples=args.max_eval_samples)
    else:
        print(f"Warning: Test data not found")

    tokenizer = femr.models.tokenizer.FEMRTokenizer.from_pretrained(args.model_path)
    batch_processor = femr.models.processor.FEMRBatchProcessor(tokenizer)
    data_collator = DataCollatorForCLMBR(batch_processor)
    
    # 2. Setup Model
    num_points = len(train_dataset.PREDICTION_POINTS)
    num_metrics = len(train_dataset.ALL_METRICS)
    
    model = CLMBRForRenji(args.model_path, num_points, num_metrics, dropout=args.dropout)
    
    # 3. Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="no", 
        save_strategy="epoch", 
        load_best_model_at_end=False,
        save_total_limit=2,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        learning_rate=args.lr,
        weight_decay=0.01,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb",
        ddp_find_unused_parameters=True,
        label_names=["labels"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        compute_metrics=None,
    )
    
    print("Starting training...")
    trainer.train()
    print(f"Training complete. Saving model to {args.output_dir}")
    trainer.save_model()
    
    # 4. Evaluate on Test Set
    if eval_dataset is not None:
        print("Running evaluation on test set...")
        # explicitly set compute_metrics for evaluation
        trainer.compute_metrics = compute_metrics
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        print("Test Set Metrics:", metrics)

if __name__ == "__main__":
    train()

# python models/clmbr/train_clmbr_renji.py
# torchrun --nproc_per_node=4 models/clmbr/train_clmbr_renji.py