import os
import sys
import pandas as pd
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import roc_auc_score
from transformers import HfArgumentParser, set_seed, Trainer, TrainingArguments

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.renji_dataset import RenjiDataset
from models.encoder_classifier import LongTableEncoderClassifier
from models.TableEncoder.config import LongTableEncoderMemoryConfig
from utils.weight_loader import load_model_weights
from utils.load_embedding import load_embedding_cache
from utils.collate import create_collate_fn

@dataclass
class ModelArguments:
    use_lora: bool = field(default=False, metadata={"help": "Set True if the checkpoint was saved with LoRA (PEFT) and adapter_config.json is absent/needs override"})
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to base transformer weights (e.g., google/tapas-base) if the model requires them before loading the classifier head/adapter."})


@dataclass
class DataArguments:
    data_dir: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/Renji")
    embedding_cache: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/.cache/embeddings/renji/text_embeddings.pt")
    checkpoint_dir: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/renji_classifier", metadata={"help": "Path to the checkpoint directory"})
    batch_size: int = field(default=64, metadata={"help": "Evaluation batch size"})
    max_table_len: Optional[int] = field(default=None, metadata={"help": "Keep only the most recent N table rows before encoding"})
    split: str = field(default="test", metadata={"help": "Dataset split to evaluate on (test/val/train)"})
    seed: int = field(default=42)
    type_vocab_file: str = field(default="data/type_vocab.json")

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    set_seed(data_args.seed)
    _, text_dim = load_embedding_cache(data_args.embedding_cache)
    
    test_dataset = RenjiDataset(
        root_dir=data_args.data_dir, split=data_args.split, table_mode="table_only", shuffle=False,
        task_mode="multi_label"
    )
    if len(test_dataset) == 0: sys.exit(0)

    vocab_path = os.path.join(project_root, data_args.type_vocab_file)
    with open(vocab_path, 'r') as f:
        type_vocab = json.load(f)

    encoder_config = LongTableEncoderMemoryConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        num_points=len(RenjiDataset.ALL_POINTS),
        num_metrics=len(RenjiDataset.ALL_METRICS),
        num_classes=len(RenjiDataset.ALL_POINTS) * len(RenjiDataset.ALL_METRICS),
        problem_type="multi_label_classification"
    )
    
    model = LongTableEncoderClassifier(config=encoder_config)

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
        data_collator=create_collate_fn(type_vocab, max_table_len=data_args.max_table_len),
    )
    
    print("Starting evaluation...")
    predict_outputs = trainer.predict(test_dataset)
    logits = predict_outputs.predictions
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid on logits
    labels_np = predict_outputs.label_ids
    
    results = [] # Store dicts: {point, metric, label, prob}
    
    for i in range(len(labels_np)):
        for p_idx, p_key in enumerate(RenjiDataset.ALL_POINTS):
            _, label_prefix, readable_point = RenjiDataset.PREDICTION_POINTS[p_key]
            for m_idx, metric in enumerate(RenjiDataset.ALL_METRICS):
                label_val = labels_np[i, p_idx, m_idx]
                
                if label_val != -100:
                    results.append({
                        'label': float(label_val),
                        'prob': float(probs[i, p_idx, m_idx]),
                        'point': readable_point,
                        'metric': metric,
                        'window': label_prefix
                    })
            
    print("\n=== Evaluation Results (AUROC Only) ===")
    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("No results collected.")
        return

    grouped = df_results.groupby(['point', 'metric'])
    print(f"{'Prediction Point':<20} | {'Metric':<10} | {'AUROC':<8} | {'N':<5} | {'Pos':<5}")
    print("-" * 65)
    
    final_output = []
    for (point, metric), group in grouped:
        y_true, y_score = group['label'].values, group['prob'].values
        n_samples, n_pos = len(y_true), sum(y_true)
        
        try: auroc = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else float('nan')
        except: auroc = float('nan')
        
        print(f"{point:<20} | {metric:<10} | {auroc:.4f}   | {n_samples:<5} | {n_pos:<5}")
        final_output.append({
            'point': point, 'metric': metric, 'auroc': auroc, 'n_samples': n_samples, 'n_pos': n_pos
        })
    
    avg_auroc = pd.DataFrame(final_output)['auroc'].mean()
    print("-" * 65)
    print(f"{'Macro Average':<20} | {'ALL':<10} | {avg_auroc:.4f}   | {len(df_results):<5} | {sum(df_results['label'])}")
    
    output_file = os.path.join(data_args.checkpoint_dir, f"test_results_{data_args.split}_auroc.csv")
    pd.DataFrame(final_output).to_csv(output_file, index=False)
    print(f"\nGrouped AUROC results saved to {output_file}")
    
    raw_file = os.path.join(data_args.checkpoint_dir, f"test_raw_predictions_{data_args.split}.csv")
    df_results.to_csv(raw_file, index=False)
    print(f"Raw predictions saved to {raw_file}")


if __name__ == "__main__":
    main()
