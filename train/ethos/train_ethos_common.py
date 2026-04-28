import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from transformers import EarlyStoppingCallback, GPT2Config, Trainer, TrainingArguments, set_seed

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))

from build_dataset_vocab import _as_meds_df, _tokenize
from ethos.vocabulary import Vocabulary
from models.ethos import GPT2NoBiasForSequenceClassification
from utils.metrics import compute_classification_metrics


def rank0_print(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", "-1")) in (-1, 0):
        print(*args, **kwargs)


def normalize_label(raw_label) -> str:
    label = str(raw_label).strip()
    while len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
        label = label[1:-1].strip()
    try:
        numeric = float(label)
        if numeric.is_integer():
            label = str(int(numeric))
    except Exception:
        pass
    lowered = label.lower()
    if lowered in {"yes", "y", "true"}:
        return "yes"
    if lowered in {"no", "n", "false"}:
        return "no"
    return lowered


def build_label_metadata(task_info: dict, task_name: str):
    info = task_info[task_name]
    task_type = info["task_type"]
    if task_type in {"multi_label_classification", "generative_task"}:
        raise NotImplementedError(f"ETHOS sequence classification does not support task '{task_name}'.")

    if "candidate" in info:
        candidates = [str(x) for x in info["candidate"]]
    elif task_type == "binary_classification":
        candidates = ["0", "1"]
    else:
        candidates = [str(i) for i in range(int(info["num_classes"]))]

    candidates = [normalize_label(x) for x in candidates]
    label_to_id = {label: i for i, label in enumerate(candidates)}
    if task_type == "binary_classification":
        if "0" in label_to_id and "1" in label_to_id:
            label_to_id.update({"no": label_to_id["0"], "false": label_to_id["0"], "n": label_to_id["0"]})
            label_to_id.update({"yes": label_to_id["1"], "true": label_to_id["1"], "y": label_to_id["1"]})
        if "no" in label_to_id and "yes" in label_to_id:
            label_to_id.update({"0": label_to_id["no"], "false": label_to_id["no"], "n": label_to_id["no"]})
            label_to_id.update({"1": label_to_id["yes"], "true": label_to_id["yes"], "y": label_to_id["yes"]})

    id_to_label = {i: label for i, label in enumerate(candidates)}
    return candidates, label_to_id, id_to_label


class EthosOnTheFlyCollator:
    def __init__(
        self,
        *,
        vocab_dir: str,
        label_to_id: dict,
        task_name: str,
        max_seq_length: int,
        static_prefixes: str = "GENDER,RACE,ETHNICITY,MARITAL",
        unknown_event_token: str = "UNKNOWN_EVENT",
        empty_context_token: str = "NO_EVENT_CONTEXT",
    ):
        vocab_dir = Path(vocab_dir)
        self.vocab = Vocabulary.from_path(vocab_dir)
        self.quantiles = json.load((vocab_dir / "quantiles.json").open())
        self.text_values = json.load((vocab_dir / "text_values.json").open())
        self.static_roots = [x.strip().upper() for x in static_prefixes.split(",") if x.strip()]
        self.label_to_id = label_to_id
        self.task_name = task_name
        self.max_seq_length = int(max_seq_length)
        self.unknown_event_token = unknown_event_token
        self.empty_context_token = empty_context_token
        self.pad_token_id = 0

    def _encode_label(self, raw_label):
        label = normalize_label(raw_label)
        if label not in self.label_to_id:
            raise ValueError(
                f"Unexpected label '{raw_label}' (normalized: '{label}') for task '{self.task_name}'. "
                f"Expected one of: {sorted(self.label_to_id.keys())}"
            )
        return self.label_to_id[label]

    def _encode_sample(self, sample):
        df = _as_meds_df(sample, 1, self.empty_context_token)
        tokenized, _, _ = _tokenize(
            df,
            self.static_roots,
            self.quantiles,
            self.text_values,
            self.vocab,
            self.unknown_event_token,
            self.empty_context_token,
        )
        ids = torch.tensor([self.vocab.stoi[token] for token in tokenized["code"]], dtype=torch.long)
        if ids.numel() > self.max_seq_length:
            ids = ids[-self.max_seq_length :]
        return ids

    def __call__(self, features):
        encoded = [self._encode_sample(feature) for feature in features]
        max_len = max(int(ids.numel()) for ids in encoded)
        input_ids, attention_mask = [], []
        for ids in encoded:
            pad_len = max_len - int(ids.numel())
            mask = torch.ones_like(ids)
            if pad_len:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros((pad_len,), dtype=mask.dtype)])
            input_ids.append(ids)
            attention_mask.append(mask)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.tensor([self._encode_label(feature["output"]) for feature in features], dtype=torch.long),
        }


@dataclass
class EthosModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Optional ETHOS checkpoint."})
    freeze_encoder: bool = field(default=False)
    n_layer: int = field(default=6)
    n_head: int = field(default=8)
    n_embd: int = field(default=512)
    dropout: float = field(default=0.1)
    classifier_dropout: Optional[float] = field(default=None)
    early_stopping_patience: int = field(default=10)
    early_stopping_threshold: float = field(default=0.0)


def build_ethos_model(model_args, vocab_dir: str, max_seq_length: int, num_labels: int, id_to_label: dict, label_to_id: dict):
    vocab = Vocabulary.from_path(vocab_dir)
    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=int(max_seq_length),
        n_embd=int(model_args.n_embd),
        n_layer=int(model_args.n_layer),
        n_head=int(model_args.n_head),
        resid_pdrop=float(model_args.dropout),
        embd_pdrop=float(model_args.dropout),
        attn_pdrop=float(model_args.dropout),
        activation_function="gelu_new",
        pad_token_id=0,
        bias=False,
    )
    config.num_labels = int(num_labels)
    config.id2label = id_to_label
    config.label2id = label_to_id
    config.problem_type = "single_label_classification"

    model = GPT2NoBiasForSequenceClassification(config, num_labels=num_labels, classifier_dropout=model_args.classifier_dropout)
    if model_args.model_name_or_path:
        state_path = Path(model_args.model_name_or_path)
        if state_path.is_dir():
            state_path = state_path / "model.safetensors"
        state = load_file(str(state_path)) if state_path.suffix == ".safetensors" else torch.load(state_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    if model_args.freeze_encoder:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    return model, vocab


def run_training(*, model_args, data_args, training_args: TrainingArguments, task_name: str, task_info: dict, train_dataset, eval_dataset):
    training_args.remove_unused_columns = False
    training_args.seed = getattr(training_args, "seed", 42) or 42
    set_seed(training_args.seed)

    metric = task_info[task_name].get("metric", "auroc").lower()
    if training_args.eval_strategy != "no" and eval_dataset is not None:
        training_args.metric_for_best_model = "eval_accuracy" if metric == "accuracy" else "eval_auroc"
        training_args.greater_is_better = True
        training_args.load_best_model_at_end = True

    candidates, label_to_id, id_to_label = build_label_metadata(task_info, task_name)
    model, vocab = build_ethos_model(
        model_args,
        vocab_dir=data_args.vocab_dir,
        max_seq_length=data_args.max_seq_length,
        num_labels=len(candidates),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
    )
    collator = EthosOnTheFlyCollator(
        vocab_dir=data_args.vocab_dir,
        label_to_id=label_to_id,
        task_name=task_name,
        max_seq_length=data_args.max_seq_length,
    )

    rank0_print("=" * 80)
    rank0_print(f"ETHOS on-the-fly train: {task_name}")
    rank0_print("=" * 80)
    rank0_print(f"Vocab dir: {data_args.vocab_dir}")
    rank0_print(f"Vocab size: {len(vocab)}")
    rank0_print(f"Train size: {len(train_dataset)}")
    if eval_dataset is not None:
        rank0_print(f"Validation size: {len(eval_dataset)}")
        rank0_print(f"Metric for best model: {training_args.metric_for_best_model}")
    rank0_print(f"ETHOS config: layers={model_args.n_layer}, heads={model_args.n_head}, hidden={model_args.n_embd}")
    rank0_print(f"Output dir: {training_args.output_dir}")

    callbacks = []
    if eval_dataset is not None and training_args.eval_strategy != "no" and model_args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=model_args.early_stopping_patience,
                early_stopping_threshold=model_args.early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        data_collator=collator,
        compute_metrics=compute_classification_metrics if eval_dataset is not None and training_args.eval_strategy != "no" else None,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)

    metadata = {
        "task_name": task_name,
        "vocab_dir": data_args.vocab_dir,
        "vocab_size": len(vocab),
        "max_seq_length": data_args.max_seq_length,
        "n_layer": model_args.n_layer,
        "n_head": model_args.n_head,
        "n_embd": model_args.n_embd,
        "freeze_encoder": model_args.freeze_encoder,
        "label2id": label_to_id,
        "id2label": {str(k): v for k, v in id_to_label.items()},
        "tokenization": "on_the_fly",
    }
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "ethos_training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
