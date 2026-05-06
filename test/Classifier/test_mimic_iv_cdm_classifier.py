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
mimic_iv_cdm_dataset_root = os.path.join(project_root, "dataset", "mimic_iv_cdm")
if mimic_iv_cdm_dataset_root not in sys.path:
    sys.path.append(mimic_iv_cdm_dataset_root)

from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from models.encoder_classifier import LongTableEncoderClassifier
from models.TableEncoder.config import LongTableEncoderMemoryConfig

from utils.collate import create_collate_fn
from utils.load_embedding import load_embedding_cache
from utils.weight_loader import load_model_weights

LABEL_MAP = {
    'appendicitis': 0,
    'cholecystitis': 1,
    'diverticulitis': 2,
    'pancreatitis': 3,
}
NUM_CLASSES = len(LABEL_MAP)

@dataclass
class ModelArguments:
    dim_out: Optional[int] = field(default=None, metadata={"help": "Classifier/encoder output dimension used by the checkpoint. Use None to keep config.dim."})
    use_lora: bool = field(default=False, metadata={"help": "Set True if the checkpoint was saved with LoRA (PEFT) and adapter_config.json is absent/needs override"})
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to base transformer weights (e.g., google/tapas-base) if the model requires them before loading the classifier head/adapter."})

@dataclass
class DataArguments:
    data_dir: str = field(default="/data/EHR_data_public/mimic-iv-cdm", metadata={"help": "Root directory for MIMIC-IV-CDM data"})
    embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings.pt",
                                  metadata={"help": "Path to pre-computed embedding cache"})
    checkpoint_dir: str = field(default=None, metadata={"help": "Path to the checkpoint directory"})
    batch_size: int = field(default=64, metadata={"help": "Evaluation batch size"})
    max_table_len: Optional[int] = field(default=None, metadata={"help": "Keep only the most recent N table rows before encoding"})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit evaluation samples"})
    task_name: str = field(default="MIMIC-IV-CDM Main Disease Diagnoses", metadata={"help": "The specific task name to test"})
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json", metadata={"help": "Path to type vocabulary JSON file"})
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
    print(f"Loading MIMIC-IV-CDM dataset from {data_args.data_dir}...")
    val_dataset = MIMICIVCDM(
        root_dir=data_args.data_dir,
        split="val",
        task_name=data_args.task_name,
        table_mode="table_only",
        lazy_mode=False,
        shuffle=False,
    )
    test_dataset = MIMICIVCDM(
        root_dir=data_args.data_dir,
        split="test",
        task_name=data_args.task_name,
        table_mode="table_only",
        lazy_mode=False,
        shuffle=False,
    )
    # Merge val into test
    test_dataset.list_data = val_dataset.list_data + test_dataset.list_data
    test_dataset.data = val_dataset.data + test_dataset.data
    print(f"Merged val ({len(val_dataset)}) + test ({len(test_dataset) - len(val_dataset)}) = {len(test_dataset)} samples")
    
    if data_args.max_eval_samples:
        test_dataset.list_data = test_dataset.list_data[:data_args.max_eval_samples]
        test_dataset.data = test_dataset.data[:data_args.max_eval_samples]
        print(f"Truncated to {len(test_dataset)} samples.")

    if len(test_dataset) == 0:
        print("Dataset is empty. Exiting.")
        sys.exit(0)

    checkpoint_config = os.path.join(data_args.checkpoint_dir, "config.json")
    if os.path.exists(checkpoint_config):
        encoder_config = LongTableEncoderMemoryConfig.from_pretrained(data_args.checkpoint_dir)
        encoder_config.text_dim = text_dim
        encoder_config.type_vocab_size = len(type_vocab)
        encoder_config.num_classes = NUM_CLASSES
        encoder_config.problem_type = "single_label_classification"
        if model_args.dim_out is not None:
            encoder_config.dim_out = model_args.dim_out
    else:
        encoder_config = LongTableEncoderMemoryConfig(
            text_dim=text_dim,
            type_vocab_size=len(type_vocab),
            dim_out=model_args.dim_out,
            num_classes=NUM_CLASSES,
            problem_type="single_label_classification"
        )
    print(f"dim_out: {encoder_config.dim_out}")

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
        data_collator=create_collate_fn(type_vocab, label_map=LABEL_MAP, max_table_len=data_args.max_table_len),
    )
    
    print("Starting evaluation...")
    predict_outputs = trainer.predict(test_dataset)
    logits = predict_outputs.predictions
    labels_np = predict_outputs.label_ids
    
    results = []
    
    all_targets = labels_np.tolist()
    all_probs = []
    all_preds = []

    # Multi-class classification for MIMIC-IV-CDM
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

    try:
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except ValueError:
        auroc = 0.5

    acc = accuracy_score(y_true, y_pred)
    
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
    
    raw_file = os.path.join(data_args.checkpoint_dir, f"test_raw_predictions.csv")
    df_results.to_csv(raw_file, index=False)
    print(f"Raw predictions saved to {raw_file}")


if __name__ == "__main__":
    main()
