import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from dataset.renji.renji_dataset import RenjiDataset
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.encoder_classifier import LongTableEncoderClassifier
from utils.collate import create_collate_fn
from utils.load_embedding import load_embedding_cache
from utils.weight_loader import load_model_weights


ACTIVE_POINTS = ["day30", "day180", "day365"]


@dataclass
class ModelArguments:
    use_lora: bool = field(default=False)
    pretrained_path: Optional[str] = field(default=None)
    dim_out: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    data_dir: str = field(default="/data/EHR_data_public/Renji")
    embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt")
    checkpoint_dir: str = field(default="/data/zikun_workspace/checkpoints/renji/no_query_classifier")
    batch_size: int = field(default=64)
    max_table_len: Optional[int] = field(default=None)
    split: str = field(default="test")
    seed: int = field(default=42)
    type_vocab_file: str = field(default="data/type_vocab.json")


def infer_dim_out(model_args: ModelArguments) -> int:
    if model_args.dim_out is not None:
        return int(model_args.dim_out)
    for path in [model_args.pretrained_path, None]:
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
    _, text_dim = load_embedding_cache(data_args.embedding_cache)

    test_dataset = RenjiDataset(
        root_dir=data_args.data_dir,
        split=data_args.split,
        table_mode="table_only",
        shuffle=False,
        target_prediction_points=ACTIVE_POINTS,
    )
    if len(test_dataset) == 0:
        sys.exit(0)

    vocab_path = os.path.join(project_root, data_args.type_vocab_file)
    with open(vocab_path, "r", encoding="utf-8") as f:
        type_vocab = json.load(f)

    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=infer_dim_out(model_args),
        num_points=len(ACTIVE_POINTS),
        num_metrics=len(RenjiDataset.ALL_METRICS),
        num_classes=len(ACTIVE_POINTS) * len(RenjiDataset.ALL_METRICS),
        problem_type="multi_label_classification",
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
            max_table_len=data_args.max_table_len,
        ),
    )

    print("Starting evaluation...")
    predict_outputs = trainer.predict(test_dataset)
    logits = predict_outputs.predictions
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels_np = predict_outputs.label_ids

    results = []
    for i in range(len(labels_np)):
        for p_idx, p_key in enumerate(ACTIVE_POINTS):
            _, label_prefix, readable_point = RenjiDataset.PREDICTION_POINTS[p_key]
            for m_idx, metric in enumerate(RenjiDataset.ALL_METRICS):
                label_val = labels_np[i, p_idx, m_idx]
                if label_val != -100:
                    results.append(
                        {
                            "label": float(label_val),
                            "prob": float(probs[i, p_idx, m_idx]),
                            "point": readable_point,
                            "metric": metric,
                            "window": label_prefix,
                        }
                    )

    print("\n=== Evaluation Results (AUROC Only) ===")
    df_results = pd.DataFrame(results)
    if df_results.empty:
        print("No results collected.")
        return

    grouped = df_results.groupby(["point", "metric"])
    print(f"{'Prediction Point':<20} | {'Metric':<10} | {'AUROC':<8} | {'N':<5} | {'Pos':<5}")
    print("-" * 65)

    final_output = []
    for (point, metric), group in grouped:
        y_true, y_score = group["label"].values, group["prob"].values
        n_samples, n_pos = len(y_true), sum(y_true)
        try:
            auroc = roc_auc_score(y_true, y_score) if len(set(y_true)) >= 2 else float("nan")
        except Exception:
            auroc = float("nan")

        print(f"{point:<20} | {metric:<10} | {auroc:.4f}   | {n_samples:<5} | {n_pos:<5}")
        final_output.append(
            {"point": point, "metric": metric, "auroc": auroc, "n_samples": n_samples, "n_pos": n_pos}
        )

    avg_auroc = pd.DataFrame(final_output)["auroc"].mean()
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
