import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
mimic_iv_cdm_dataset_root = os.path.join(project_root, "dataset", "mimic_iv_cdm")
if mimic_iv_cdm_dataset_root not in sys.path:
    sys.path.append(mimic_iv_cdm_dataset_root)

from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.encoder_classifier import LongTableEncoderClassifier
from utils.collate import create_collate_fn
from utils.load_embedding import load_embedding_cache
from utils.weight_loader import load_model_weights


LABEL_MAP = {
    "appendicitis": 0,
    "cholecystitis": 1,
    "diverticulitis": 2,
    "pancreatitis": 3,
}
NUM_CLASSES = len(LABEL_MAP)


@dataclass
class ModelArguments:
    use_lora: bool = field(default=False)
    pretrained_path: Optional[str] = field(default=None)
    dim_out: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_dir: str = field(default="/data/EHR_data_public/mimic-iv-cdm")
    embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt")
    checkpoint_dir: str = field(default=None)
    batch_size: int = field(default=64)
    max_table_len: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    task_name: str = field(default="MIMIC-IV-CDM Main Disease Diagnoses")
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    seed: int = field(default=42)


def infer_dim_out(model_args: ModelArguments) -> int:
    if model_args.dim_out is not None:
        return int(model_args.dim_out)
    for path in [model_args.pretrained_path]:
        if path:
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if config.get("dim_out") is not None:
                    return int(config["dim_out"])
    return 2048


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

    _, text_dim = load_embedding_cache(data_args.embedding_cache)

    with open(data_args.type_vocab_file, "r", encoding="utf-8") as f:
        type_vocab = json.load(f)

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
    test_dataset.list_data = val_dataset.list_data + test_dataset.list_data
    test_dataset.data = val_dataset.data + test_dataset.data
    print(f"Merged val ({len(val_dataset)}) + test ({len(test_dataset) - len(val_dataset)}) = {len(test_dataset)} samples")

    if data_args.max_eval_samples:
        test_dataset.list_data = test_dataset.list_data[: data_args.max_eval_samples]
        test_dataset.data = test_dataset.data[: data_args.max_eval_samples]
        print(f"Truncated to {len(test_dataset)} samples.")

    if len(test_dataset) == 0:
        print("Dataset is empty. Exiting.")
        sys.exit(0)

    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=infer_dim_out(model_args),
        num_classes=NUM_CLASSES,
        problem_type="single_label_classification",
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
        data_collator=create_collate_fn(
            type_vocab=type_vocab,
            label_map=LABEL_MAP,
            max_table_len=data_args.max_table_len,
        ),
    )

    print("Starting evaluation...")
    predict_outputs = trainer.predict(test_dataset)
    logits = predict_outputs.predictions
    labels_np = predict_outputs.label_ids

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    preds = np.argmax(probs, axis=-1)

    results = []
    for i in range(len(labels_np)):
        results.append(
            {
                "label": int(labels_np[i]),
                "prob": probs[i].tolist(),
                "pred": int(preds[i]),
                "task": data_args.task_name,
            }
        )

    print(f"\n=== Evaluation Results for {data_args.task_name} ===")
    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("No results collected.")
        return

    y_true = labels_np
    y_prob = probs
    y_pred = preds

    try:
        auroc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except ValueError:
        auroc = 0.5
    acc = accuracy_score(y_true, y_pred)

    print(f"{'Task':<20} | {'AUROC':<8} | {'Accuracy':<8} | {'N':<5}")
    print("-" * 55)
    print(f"{data_args.task_name[:20]:<20} | {auroc:.4f}   | {acc:.4f}   | {len(y_true):<5}")

    final_output = [{"task": data_args.task_name, "auroc": auroc, "accuracy": acc, "n_samples": len(y_true)}]
    output_file = os.path.join(data_args.checkpoint_dir, "test_results_metrics.csv")
    pd.DataFrame(final_output).to_csv(output_file, index=False)
    print(f"\nMetrics saved to {output_file}")

    raw_file = os.path.join(data_args.checkpoint_dir, "test_raw_predictions.csv")
    df_results.to_csv(raw_file, index=False)
    print(f"Raw predictions saved to {raw_file}")


if __name__ == "__main__":
    main()
