import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.renji.renji_dataset import RenjiDataset
from train.Llama.train_ehrshot_llama import (
    HeadOnlySequenceClassificationTrainer,
    _copy_tokenizer_config_to_output,
    _load_clmbr_tokenizer,
    _save_training_metadata,
    rank0_print,
)


def _parse_csv_arg(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def _label_names():
    return [
        f"{point_key}/{metric}"
        for point_key in RenjiDataset.ALL_POINTS
        for metric in RenjiDataset.ALL_METRICS
    ]


def _build_label_metadata():
    names = _label_names()
    label_to_id = {name: idx for idx, name in enumerate(names)}
    id_to_label = {idx: name for idx, name in enumerate(names)}
    return names, label_to_id, id_to_label


class RenjiMEDSDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        target_prediction_points=None,
        shuffle: bool = False,
        max_samples: Optional[int] = None,
    ):
        self.source = RenjiDataset(
            root_dir=root_dir,
            split=split,
            max_samples=None,
            table_mode="text_only",
            target_prediction_points=target_prediction_points,
            shuffle=False,
            task_mode="multi_label",
            return_meds=True,
        )
        self.samples = [sample for sample in self.source.samples if sample["metric"] == "all"]
        if shuffle:
            random.Random(42).shuffle(self.samples)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def _label_tensor(self, sample):
        labels = torch.full(
            (len(RenjiDataset.ALL_POINTS), len(RenjiDataset.ALL_METRICS)),
            -100,
            dtype=torch.float32,
        )
        patient_labels = self.source.labels_df.loc[sample["fname_key"]]
        for p_idx, p_key in enumerate(RenjiDataset.ALL_POINTS):
            _, prefix, _ = RenjiDataset.PREDICTION_POINTS[p_key]
            for m_idx, metric in enumerate(RenjiDataset.ALL_METRICS):
                col_name = f"{prefix}_{metric}"
                if col_name in patient_labels and pd.notna(patient_labels[col_name]):
                    labels[p_idx, m_idx] = float(patient_labels[col_name])
        return labels.reshape(-1)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        df_followup = self.source._load_followup_data(sample)
        first_row = df_followup.iloc[0]
        surgery_date = pd.to_datetime(first_row["报告日期"]) - pd.Timedelta(days=float(first_row["术后天数"]))
        static_features = self.source._get_static_features(sample["fname_key"])
        _, meds_events, hf_ehr_events = self.source.meds_input_process(
            subject_id=sample["fname_key"],
            static_features=static_features,
            df_followup=df_followup,
            surgery_date=surgery_date,
        )
        return {
            "idx": idx,
            "hf_ehr_events": hf_ehr_events,
            "meds_events": meds_events,
            "labels": self._label_tensor(sample),
            "task_info": self.source.task_schema["multi_label_prediction"],
        }


class RenjiMEDSDataCollator:
    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    @staticmethod
    def _squeeze_single_batch(tokenized):
        row = {}
        for key, value in tokenized.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().tolist()
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                row[key] = value[0]
            else:
                row[key] = value
        return row

    def _tokenize_sample(self, sample: dict):
        tokenized = self.tokenizer(
            [sample["hf_ehr_events"]],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None,
        )
        return self._squeeze_single_batch(tokenized)

    def __call__(self, features):
        encoded_rows = [self._tokenize_sample(feature) for feature in features]
        labels = torch.stack([feature["labels"] for feature in features]).float()

        batch = self.tokenizer.pad(encoded_rows, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


class RenjiLlamaMEDSMultiLabelClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        id_to_label: dict,
        label_to_id: dict,
        freeze_encoder: bool = False,
        tokenizer_vocab_size: Optional[int] = None,
        use_peft: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.use_peft = use_peft
        self.encoder = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.encoder.config.use_cache = False
        self.num_labels = num_labels

        if tokenizer_vocab_size is not None:
            current_vocab_size = int(self.encoder.get_input_embeddings().num_embeddings)
            if int(tokenizer_vocab_size) != current_vocab_size:
                self.encoder.resize_token_embeddings(int(tokenizer_vocab_size))
                rank0_print(
                    f"Resized token embeddings: {current_vocab_size} -> {int(tokenizer_vocab_size)}"
                )

        if use_peft:
            from peft import LoraConfig, get_peft_model

            target_modules = [module.strip() for module in lora_target_modules.split(",") if module.strip()]
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

        hidden_size = int(self.encoder.config.hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

        if freeze_encoder and not use_peft:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.config = self.encoder.config
        self.config.num_labels = num_labels
        self.config.id2label = id_to_label
        self.config.label2id = label_to_id
        self.config.problem_type = "multi_label_classification"

    @staticmethod
    def _pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        seq_lens = attention_mask.long().sum(dim=1) - 1
        seq_lens = torch.clamp(seq_lens, min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, seq_lens, :]

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if self.use_peft:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
                **kwargs,
            )
            hidden_states = outputs.hidden_states[-1]
        else:
            outputs = self.encoder.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                **kwargs,
            )
            hidden_states = outputs.last_hidden_state

        pooled = self._pool_last_token(hidden_states, attention_mask)
        pooled = pooled.to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            mask = labels != -100
            safe_labels = labels.clone()
            safe_labels[~mask] = 0
            loss_matrix = F.binary_cross_entropy_with_logits(
                logits,
                safe_labels.to(logits.dtype),
                reduction="none",
            )
            loss = (loss_matrix * mask.to(logits.dtype)).sum() / mask.to(logits.dtype).sum()

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


@dataclass
class ModelArguments:
    tokenizer_config_path: str = field(
        metadata={"help": "Path to tokenizer_config.json or directory containing it."},
    )
    model_name_or_path: str = field(
        default="/data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr",
        metadata={"help": "Path to StanfordShahLab llama-base-4096-clmbr weights."},
    )
    freeze_encoder: bool = field(default=True)
    use_peft: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/EHR_data_public/Renji")
    target_prediction_points: str = field(
        default="day0,day30,day180,day365",
        metadata={"help": "Comma-separated Renji prediction points, e.g. day30,day180."},
    )
    train_split: str = field(default="train")
    max_train_samples: Optional[int] = field(default=None)
    max_seq_length: int = field(default=4096)


def _load_source_dataset(data_args, split_name: str, shuffle: bool, max_samples: Optional[int]):
    dataset = RenjiMEDSDataset(
        root_dir=data_args.root_dir,
        split=split_name,
        target_prediction_points=_parse_csv_arg(data_args.target_prediction_points),
        shuffle=shuffle,
        max_samples=max_samples,
    )
    rank0_print(f"{split_name} source [Renji multi-label, MEDS] size: {len(dataset)}")
    return dataset


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    training_args.save_safetensors = True
    training_args.bf16 = True
    training_args.fp16 = False
    set_seed(training_args.seed)

    if (not model_args.freeze_encoder) and (not model_args.use_peft):
        raise ValueError(
            "Full encoder fine-tuning is disabled for this script. "
            "Use --use_peft True when setting --freeze_encoder False."
        )

    training_args.eval_strategy = "no"
    training_args.load_best_model_at_end = False

    label_names, label_to_id, id_to_label = _build_label_metadata()
    tokenizer_source = model_args.tokenizer_config_path
    tokenizer = _load_clmbr_tokenizer(tokenizer_source)

    rank0_print("=" * 80)
    rank0_print("Renji MEDS Llama Encoder Multi-Label Train")
    rank0_print("=" * 80)
    rank0_print(f"Model path: {model_args.model_name_or_path}")
    rank0_print(f"Tokenizer source: {tokenizer_source}")
    rank0_print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    rank0_print(f"Root dir: {data_args.root_dir}")
    rank0_print(f"Train split: {data_args.train_split}")
    rank0_print(f"Target prediction points: {data_args.target_prediction_points}")
    rank0_print(f"Num labels: {len(label_names)}")
    rank0_print(f"Max seq length: {data_args.max_seq_length}")
    rank0_print(f"Freeze encoder: {model_args.freeze_encoder}")
    rank0_print(f"Use PEFT: {model_args.use_peft}")
    rank0_print(f"Output dir: {training_args.output_dir}")

    model = RenjiLlamaMEDSMultiLabelClassifier(
        model_name_or_path=model_args.model_name_or_path,
        num_labels=len(label_names),
        id_to_label=id_to_label,
        label_to_id=label_to_id,
        freeze_encoder=model_args.freeze_encoder,
        tokenizer_vocab_size=int(tokenizer.vocab_size),
        use_peft=model_args.use_peft,
        lora_r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        lora_target_modules=model_args.lora_target_modules,
    )

    trainable_parameters = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if model_args.use_peft:
        encoder_trainable = [name for name in trainable_parameters if name.startswith("encoder.")]
        if not encoder_trainable:
            raise ValueError("PEFT mode expects trainable encoder adapter parameters, but none were found.")
    elif model_args.freeze_encoder:
        invalid_trainable = [name for name in trainable_parameters if not name.startswith("classifier.")]
        if invalid_trainable:
            raise ValueError(
                "Head-only training requires only classifier parameters to be trainable, "
                f"but found non-classifier trainable parameters: {invalid_trainable[:5]}"
            )
    rank0_print(f"Trainable parameter tensors: {len(trainable_parameters)}")

    train_dataset = _load_source_dataset(
        data_args=data_args,
        split_name=data_args.train_split,
        shuffle=True,
        max_samples=data_args.max_train_samples,
    )

    data_collator = RenjiMEDSDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )

    trainer = HeadOnlySequenceClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    rank0_print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    rank0_print(f"Saving checkpoint to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    _copy_tokenizer_config_to_output(tokenizer, training_args.output_dir)
    _save_training_metadata(
        output_dir=training_args.output_dir,
        model_args=model_args,
        data_args=data_args,
        tokenizer=tokenizer,
        task_name="renji_multi_label_prediction",
    )


if __name__ == "__main__":
    main()
