import os
import sys
import json
import csv
import random
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.mimic.mimic_dataset import MIMICIV
from models.TableEncoder.config import TableEncoderConfig
from models.TableEncoder.encoder import LongTableEncoder
from utils.load_embedding import load_embedding_cache
from utils.collate import build_table_token_tensors

TABLE_PLACEHOLDER_TOKEN = "<table>"


def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in (-1, 0):
        print(*args, **kwargs)


@dataclass
class ModelArguments:
    llm_path: str = field(default="/data/model_weights_public/BlueZeros/EHR-R1-1.7B")
    table_encoder_path: Optional[str] = field(default=None)
    attention_mode: str = field(default="1d")
    projector_hidden_size: int = field(default=2048)
    freeze_llm: bool = field(default=True)
    freeze_table_encoder: bool = field(default=False)
    table_to_text_weight: float = field(default=1.0)
    text_to_table_weight: float = field(default=1.0)
    table_to_text_prompt: str = field(
        default=(
            "Task: Convert table embedding to EHR markdown text.\n"
            "Use the same style as the few-shot examples below.\n"
            "\n"
            "Current sample:\n"
            "[Table Embedding]\n"
            "<table>\n"
            "\n"
            "Formatting rules:\n"
            "1. Event block title format: ## <Event Name> [<start_time>]\n"
            "2. Keep field names and casing exactly as examples.\n"
            "3. Do not mix formats across categories.\n"
            "\n"
            "Example A (markdown table style):\n"
            "## Laboratory Test Events [2180-06-27 05:10:00]\n"
            "| Item Name | Value | Unit | Range | Flag | Comments |\n"
            "| ------ | ------ | ------ | ------ | ------ | ------ |\n"
            "| Lactate | 1.5 | mmol/L | 0.9-1.1 | abnormal | |\n"
            "\n"
            "Example B (key-value bullet style):\n"
            "## Vitalsign [2180-06-27 08:10:00]\n"
            "- Temperature: 98.6\n"
            "- Heartrate: 88\n"
            "- Resprate: 18\n"
            "- O2sat: 98\n"
            "- Sbp: 118\n"
            "- Dbp: 72\n"
            "- Pain: 0\n"
            "- Rhythm: Normal sinus rhythm\n"
            "\n"
            "Example C (simple bullet style):\n"
            "## Electronic Medicine Administration Record [2180-05-07 00:44:00]\n"
            "- Potassium Chloride\n"
            "\n"
            "Now generate markdown text for the current sample.\n"
            "Output:\n"
        )
    )
    text_to_table_prompt: str = field(
        default=(
            "Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.\n"
            "Output JSON list only.\n"
            "Each element: {\"Item\": str, \"Value\": str}\n"
            "\n"
            "Current sample template:\n"
            "Markdown Content:\n"
            "<MARKDOWN_CONTENT>\n"
            "\n"
            "[Table Embedding]\n"
            "<table>\n"
            "\n"
            "Example A (markdown table style):\n"
            "Markdown Content:\n"
            "## Laboratory Test Events [2180-06-27 05:10:00]\n"
            "| Item Name | Value | Unit | Range | Flag | Comments |\n"
            "| ------ | ------ | ------ | ------ | ------ | ------ |\n"
            "| Lactate | 1.5 | mmol/L | 0.9-1.1 | abnormal | |\n"
            "\n"
            "[Table Embedding]\n"
            "<table>\n"
            "\n"
            "Output:\n"
            "[{\"Item\":\"Lactate\",\"Value\":\"1.5\"}]\n"
            "\n"
            "Example B (key-value bullet style):\n"
            "Markdown Content:\n"
            "## Vitalsign [2180-06-27 08:10:00]\n"
            "- Temperature: 98.6\n"
            "- Heartrate: 88\n"
            "- Resprate: 18\n"
            "\n"
            "[Table Embedding]\n"
            "<table>\n"
            "\n"
            "Output:\n"
            "[{\"Item\":\"Heartrate\",\"Value\":\"88\"}]\n"
            "\n"
            "Example C (simple bullet style):\n"
            "Markdown Content:\n"
            "## Electronic Medicine Administration Record [2180-05-07 00:44:00]\n"
            "- Potassium Chloride\n"
            "\n"
            "[Table Embedding]\n"
            "<table>\n"
            "\n"
            "Output:\n"
            "[{\"Item\":\"Potassium Chloride\",\"Value\":\"Potassium Chloride\"}]\n"
            "\n"
            "Now output reconstructed JSON for current sample.\n"
            "Output:\n"
        )
    )


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    train_sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/bi_reconstruct.csv"
    )
    type_vocab_file: str = field(default="/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/data/type_vocab.json")
    table_text_embedding: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/embeddings/table_text_embeddings.pt"
    )
    random_subset_size: Optional[int] = field(default=None)
    random_subset_seed: int = field(default=42)
    max_train_samples: Optional[int] = field(default=None)
    max_target_length: int = field(default=1024)
    mask_ratio: float = field(default=0.15)
    max_masked_cells: int = field(default=64)
    lazy_mode: bool = field(default=True)


@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    output_dir: str = field(default="/data/zikun_workspace/checkpoints/pretraining/stage2_bi_reconstruct")
    per_device_train_batch_size: int = field(default=2)
    learning_rate: float = field(default=3e-5)
    num_train_epochs: float = field(default=1.0)
    warmup_ratio: float = field(default=0.03)
    weight_decay: float = field(default=0.01)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    save_total_limit: int = field(default=2)
    bf16: bool = field(default=True)
    disable_tqdm: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_project: Optional[str] = field(default="MIMIC-BiReconstruct")
    save_encoder_only: bool = field(default=False)


class BiReconstructCollator:
    def __init__(
        self,
        tokenizer,
        text_to_idx: Dict[str, int],
        type_vocab: Dict[str, int],
        max_target_length: int,
        mask_ratio: float,
        max_masked_cells: int,
        seed: int,
    ):
        self.tokenizer = tokenizer
        self.text_to_idx = text_to_idx
        self.type_vocab = type_vocab
        self.max_target_length = max_target_length
        self.mask_ratio = mask_ratio
        self.max_masked_cells = max_masked_cells
        self.seed = seed

        self.pad_idx = self.text_to_idx["[PAD]"]

    def _mask_table(self, df: pd.DataFrame, sample_uid: str) -> Tuple[pd.DataFrame, str]:
        work_df = df.reset_index(drop=True).copy()

        candidates: List[Tuple[int, str]] = []
        for row_id in range(len(work_df)):
            item = str(work_df.at[row_id, "Item"]).strip()
            value = str(work_df.at[row_id, "Value"]).strip()
            if value and value.lower() != "nan":
                candidates.append((row_id, "Value"))
            elif item and item.lower() != "nan":
                candidates.append((row_id, "Item"))

        rng = random.Random((hash(sample_uid) ^ self.seed) & 0xFFFFFFFF)
        num_mask = int(len(candidates) * self.mask_ratio)
        num_mask = max(1, min(num_mask, self.max_masked_cells))
        chosen = rng.sample(candidates, num_mask)

        target = []
        for row_id, masked_col in chosen:
            item_name = str(work_df.at[row_id, "Item"])
            value_text = str(work_df.at[row_id, "Value"]).strip()
            if not value_text or value_text.lower() == "nan":
                value_text = item_name
            target.append({"Item": item_name, "Value": value_text})
            work_df.at[row_id, masked_col] = "[EMPTY]"

        return work_df, json.dumps(target, ensure_ascii=False)

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        full_tables = []
        masked_tables = []
        table_to_text_targets = []
        text_to_table_targets = []
        text_to_table_prompts = []

        for sample in batch:
            table_df = sample["measurement_table"]
            sample_uid = str(sample["idx"])

            full_tables.append(table_df.reset_index(drop=True))
            masked_df, masked_target = self._mask_table(table_df, sample_uid)
            masked_tables.append(masked_df)

            table_to_text_targets.append(sample["input"])
            text_to_table_targets.append(masked_target)
            text_to_table_prompts.append(
                "Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.\n"
                "Markdown Content is fully observed and does not contain [EMPTY].\n"
                "The masked positions exist only in the table embedding.\n"
                "Output JSON list only.\n"
                "Each element: {\"Item\": str, \"Value\": str}\n"
                "\n"
                "[Table Embedding]\n"
                f"{TABLE_PLACEHOLDER_TOKEN}\n\n"
                "Markdown Content:\n"
                f"{sample['input']}\n\n"
                "Output:\n"
            )

        full_batch = build_table_token_tensors(
            tables_list=full_tables,
            text_to_idx=self.text_to_idx,
            pad_idx=self.pad_idx,
            type_vocab=self.type_vocab,
        )
        masked_batch = build_table_token_tensors(
            tables_list=masked_tables,
            text_to_idx=self.text_to_idx,
            pad_idx=self.pad_idx,
            type_vocab=self.type_vocab,
        )

        t2t_tokens = self.tokenizer(
            table_to_text_targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        x2t_tokens = self.tokenizer(
            text_to_table_targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        x2t_prompt_tokens = self.tokenizer(
            text_to_table_prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )

        return {
            "item_ids": full_batch["item_ids"],
            "unit_ids": full_batch["unit_ids"],
            "value_text_ids": full_batch["value_text_ids"],
            "times": full_batch["times"],
            "numeric_values": full_batch["numeric_values"],
            "numeric_mask": full_batch["numeric_mask"],
            "type_ids": full_batch["type_ids"],
            "seq_mask": full_batch["seq_mask"],
            "masked_item_ids": masked_batch["item_ids"],
            "masked_unit_ids": masked_batch["unit_ids"],
            "masked_value_text_ids": masked_batch["value_text_ids"],
            "masked_times": masked_batch["times"],
            "masked_numeric_values": masked_batch["numeric_values"],
            "masked_numeric_mask": masked_batch["numeric_mask"],
            "masked_type_ids": masked_batch["type_ids"],
            "masked_seq_mask": masked_batch["seq_mask"],
            "table_to_text_target_ids": t2t_tokens["input_ids"],
            "table_to_text_target_mask": t2t_tokens["attention_mask"],
            "text_to_table_target_ids": x2t_tokens["input_ids"],
            "text_to_table_target_mask": x2t_tokens["attention_mask"],
            "text_to_table_prompt_ids": x2t_prompt_tokens["input_ids"],
            "text_to_table_prompt_mask": x2t_prompt_tokens["attention_mask"],
            "labels": torch.zeros(len(batch), dtype=torch.long),
        }


class BiReconstructModel(nn.Module):
    def __init__(
        self,
        table_encoder: LongTableEncoder,
        embedding_matrix: torch.Tensor,
        llm,
        tokenizer,
        table_to_text_prompt: str,
        table_placeholder_token_id: int,
        table_to_text_weight: float,
        text_to_table_weight: float,
    ):
        super().__init__()
        self.table_encoder = table_encoder
        self.table_token_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.llm = llm

        self.table_to_text_weight = table_to_text_weight
        self.text_to_table_weight = text_to_table_weight

        self.register_buffer(
            "table_to_text_prompt_ids",
            torch.tensor(tokenizer(table_to_text_prompt, add_special_tokens=False)["input_ids"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "table_placeholder_token_id",
            torch.tensor(table_placeholder_token_id, dtype=torch.long),
            persistent=False,
        )

    def _encode_table(
        self,
        item_ids,
        unit_ids,
        value_text_ids,
        times,
        numeric_values,
        numeric_mask,
        type_ids,
        seq_mask,
    ):
        table_repr = self.table_encoder(
            item_emb=self.table_token_embedding(item_ids),
            unit_emb=self.table_token_embedding(unit_ids),
            value_emb=self.table_token_embedding(value_text_ids),
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            type_ids=type_ids,
            seq_mask=seq_mask,
        )
        return table_repr

    def _build_decoder_inputs(
        self,
        table_embeds: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = table_embeds.device
        embed_layer = self.llm.get_input_embeddings()
        model_dtype = embed_layer.weight.dtype

        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        table_embeds = table_embeds.to(dtype=model_dtype)

        inputs_list = []
        labels_list = []
        attn_list = []

        for b in range(table_embeds.size(0)):
            valid_prompt = prompt_ids[b][prompt_mask[b].bool()].to(device)
            valid_target = target_ids[b][target_mask[b].bool()].to(device)
            placeholder_pos = int((valid_prompt == self.table_placeholder_token_id).nonzero(as_tuple=False).squeeze(-1)[0].item())
            pre_ids = valid_prompt[:placeholder_pos]
            post_ids = valid_prompt[placeholder_pos + 1:]
            target_input_ids = valid_target[:-1] if valid_target.numel() > 0 else valid_target

            embed_parts = []
            if pre_ids.numel() > 0:
                embed_parts.append(embed_layer(pre_ids).to(dtype=model_dtype))
            embed_parts.append(table_embeds[b])
            if post_ids.numel() > 0:
                embed_parts.append(embed_layer(post_ids).to(dtype=model_dtype))
            if target_input_ids.numel() > 0:
                embed_parts.append(embed_layer(target_input_ids).to(dtype=model_dtype))
            full_input = torch.cat(embed_parts, dim=0)

            seq_tokens = (
                pre_ids.tolist()
                + [None] * table_embeds[b].size(0)
                + post_ids.tolist()
                + valid_target.tolist()
            )
            full_label = torch.tensor(
                [-100 if tok is None else tok for tok in seq_tokens[1:]],
                device=device,
                dtype=torch.long,
            )
            full_attn = torch.ones(full_input.size(0), device=device, dtype=torch.long)

            inputs_list.append(full_input)
            labels_list.append(full_label)
            attn_list.append(full_attn)

        max_len = max(x.size(0) for x in inputs_list)
        hidden_dim = inputs_list[0].size(1)

        inputs = torch.zeros(len(inputs_list), max_len, hidden_dim, device=device, dtype=model_dtype)
        labels = torch.full((len(inputs_list), max_len), -100, device=device, dtype=torch.long)
        attention_mask = torch.zeros(len(inputs_list), max_len, device=device, dtype=torch.long)

        for i in range(len(inputs_list)):
            cur_len = inputs_list[i].size(0)
            inputs[i, :cur_len] = inputs_list[i]
            labels[i, :cur_len] = labels_list[i]
            attention_mask[i, :cur_len] = attn_list[i]

        return inputs, attention_mask, labels

    def _task_loss(
        self,
        table_embeds: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds, attention_mask, labels = self._build_decoder_inputs(
            table_embeds=table_embeds,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            target_ids=target_ids,
            target_mask=target_mask,
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return outputs.loss

    def forward(
        self,
        item_ids,
        unit_ids,
        value_text_ids,
        times,
        numeric_values,
        numeric_mask,
        type_ids,
        seq_mask,
        masked_item_ids,
        masked_unit_ids,
        masked_value_text_ids,
        masked_times,
        masked_numeric_values,
        masked_numeric_mask,
        masked_type_ids,
        masked_seq_mask,
        table_to_text_target_ids,
        table_to_text_target_mask,
        text_to_table_target_ids,
        text_to_table_target_mask,
        text_to_table_prompt_ids,
        text_to_table_prompt_mask,
        labels=None,
        **kwargs,
    ):
        full_prefix = self._encode_table(
            item_ids=item_ids,
            unit_ids=unit_ids,
            value_text_ids=value_text_ids,
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            type_ids=type_ids,
            seq_mask=seq_mask,
        )
        masked_prefix = self._encode_table(
            item_ids=masked_item_ids,
            unit_ids=masked_unit_ids,
            value_text_ids=masked_value_text_ids,
            times=masked_times,
            numeric_values=masked_numeric_values,
            numeric_mask=masked_numeric_mask,
            type_ids=masked_type_ids,
            seq_mask=masked_seq_mask,
        )

        bs = full_prefix.size(0)
        t2t_prompt_ids = self.table_to_text_prompt_ids.unsqueeze(0).expand(bs, -1)
        t2t_prompt_mask = torch.ones_like(t2t_prompt_ids)

        loss_t2t = self._task_loss(
            table_embeds=full_prefix,
            prompt_ids=t2t_prompt_ids,
            prompt_mask=t2t_prompt_mask,
            target_ids=table_to_text_target_ids,
            target_mask=table_to_text_target_mask,
        )
        loss_x2t = self._task_loss(
            table_embeds=masked_prefix,
            prompt_ids=text_to_table_prompt_ids,
            prompt_mask=text_to_table_prompt_mask,
            target_ids=text_to_table_target_ids,
            target_mask=text_to_table_target_mask,
        )

        loss = self.table_to_text_weight * loss_t2t + self.text_to_table_weight * loss_x2t
        return {
            "loss": loss,
            "loss_table_to_text": loss_t2t.detach(),
            "loss_text_to_table": loss_x2t.detach(),
        }


def _prepare_random_subset_csv(source_path: str, subset_size: Optional[int], seed: int) -> str:
    if subset_size is None:
        return source_path

    source_path = os.path.abspath(source_path)
    stat = os.stat(source_path)
    fingerprint = hashlib.md5(
        f"{source_path}|{stat.st_size}|{stat.st_mtime_ns}|{subset_size}|{seed}".encode("utf-8")
    ).hexdigest()[:12]
    subset_dir = os.path.join(project_root, ".cache", "bi_reconstruct_subsets")
    os.makedirs(subset_dir, exist_ok=True)
    subset_path = os.path.join(
        subset_dir,
        f"{os.path.splitext(os.path.basename(source_path))[0]}_sample{subset_size}_{fingerprint}.csv",
    )

    if os.path.exists(subset_path):
        rank0_print(f"Using cached random subset CSV: {subset_path}")
        return subset_path

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    temp_path = subset_path + ".tmp"

    if local_rank in (-1, 0):
        rank0_print(
            f"Creating random subset CSV from {source_path} "
            f"(subset_size={subset_size}, seed={seed})"
        )
        rng = random.Random(seed)
        sampled_rows = []
        total_rows = 0

        with open(source_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise ValueError(f"No CSV header found in {source_path}")

            for row in reader:
                total_rows += 1
                if len(sampled_rows) < subset_size:
                    sampled_rows.append(row)
                    continue

                replace_at = rng.randint(0, total_rows - 1)
                if replace_at < subset_size:
                    sampled_rows[replace_at] = row

        if total_rows == 0:
            raise ValueError(f"No rows found in {source_path}")

        with open(temp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_rows)
        os.replace(temp_path, subset_path)
        rank0_print(f"Wrote random subset CSV with {len(sampled_rows)} rows to {subset_path}")
        return subset_path

    wait_seconds = int(os.environ.get("BI_RECONSTRUCT_SUBSET_WAIT_SECONDS", "7200"))
    poll_seconds = float(os.environ.get("BI_RECONSTRUCT_SUBSET_POLL_SECONDS", "2"))
    deadline = time.time() + max(1, wait_seconds)
    while time.time() < deadline:
        if os.path.exists(subset_path):
            return subset_path
        time.sleep(max(0.1, poll_seconds))

    raise TimeoutError(f"Timed out waiting for random subset CSV: {subset_path}")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsCustom))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("========== Stage 2: Bi-Reconstruct ==========")
    rank0_print(f"Output dir: {training_args.output_dir}")
    train_sample_info_path = _prepare_random_subset_csv(
        data_args.train_sample_info_path,
        data_args.random_subset_size,
        data_args.random_subset_seed,
    )
    rank0_print(f"Train sample info: {train_sample_info_path}")
    rank0_print(f"LLM path: {model_args.llm_path}")
    set_seed(training_args.seed)
    if training_args.run_project:
        os.environ["WANDB_PROJECT"] = training_args.run_project

    rank0_print(f"Loading type vocab from {data_args.type_vocab_file}")
    with open(data_args.type_vocab_file, "r", encoding="utf-8") as f:
        type_vocab = json.load(f)
    rank0_print(f"Loaded type vocab with {len(type_vocab)} entries")

    rank0_print(f"Loading table text embeddings from {data_args.table_text_embedding}")
    embedding_map, text_dim = load_embedding_cache(data_args.table_text_embedding)
    vocab_keys = list(embedding_map.keys())
    text_to_idx = {k: i for i, k in enumerate(vocab_keys)}
    embedding_matrix = torch.stack([embedding_map[k] for k in vocab_keys])
    rank0_print(f"Loaded table token embedding matrix: {embedding_matrix.shape}")

    rank0_print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_path, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [TABLE_PLACEHOLDER_TOKEN]})
    table_placeholder_token_id = tokenizer.convert_tokens_to_ids(TABLE_PLACEHOLDER_TOKEN)
    rank0_print(f"Tokenizer ready; placeholder token id = {table_placeholder_token_id}")

    rank0_print("Loading LLM")
    llm = AutoModelForCausalLM.from_pretrained(
        model_args.llm_path,
        trust_remote_code=True,
        dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    llm.resize_token_embeddings(len(tokenizer))
    rank0_print("LLM loaded")

    if model_args.freeze_llm:
        for p in llm.parameters():
            p.requires_grad = False
        rank0_print("LLM parameters frozen")

    rank0_print("Building table encoder")
    encoder_cfg = TableEncoderConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        attention_mode=model_args.attention_mode,
        dim_out=model_args.projector_hidden_size,
    )
    table_encoder = LongTableEncoder(config=encoder_cfg)

    if model_args.table_encoder_path:
        rank0_print(f"Loading table encoder weights from {model_args.table_encoder_path}")
        if model_args.table_encoder_path.endswith(".safetensors"):
            state = load_file(model_args.table_encoder_path)
        else:
            state = torch.load(model_args.table_encoder_path, map_location="cpu")
        table_encoder.load_state_dict(state)
        rank0_print("Loaded table encoder weights")

    if model_args.freeze_table_encoder:
        for p in table_encoder.parameters():
            p.requires_grad = False
        rank0_print("Table encoder parameters frozen")

    rank0_print("Building joint model")
    model = BiReconstructModel(
        table_encoder=table_encoder,
        embedding_matrix=embedding_matrix,
        llm=llm,
        tokenizer=tokenizer,
        table_to_text_prompt=model_args.table_to_text_prompt,
        table_placeholder_token_id=table_placeholder_token_id,
        table_to_text_weight=model_args.table_to_text_weight,
        text_to_table_weight=model_args.text_to_table_weight,
    )

    rank0_print("Building train dataset (table_length cache enabled; max_train_samples applied after sorting)")
    train_dataset = MIMICIV(
        root_dir=data_args.root_dir,
        sample_info_path=train_sample_info_path,
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode="table_only",
        max_samples=None,
        use_table_length_cache=True,
    )

    if data_args.max_train_samples is not None:
        train_dataset.sample_info = train_dataset.sample_info[: data_args.max_train_samples]


    rank0_print(f"train_dataset={len(train_dataset)}")

    rank0_print("Building collator")
    collate_fn = BiReconstructCollator(
        tokenizer=tokenizer,
        text_to_idx=text_to_idx,
        type_vocab=type_vocab,
        max_target_length=data_args.max_target_length,
        mask_ratio=data_args.mask_ratio,
        max_masked_cells=data_args.max_masked_cells,
        seed=training_args.seed,
    )

    rank0_print("Building Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )

    rank0_print("Starting trainer.train()")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if training_args.save_encoder_only:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank in (-1, 0):
            model_to_save = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            save_dir = os.path.join(training_args.output_dir, "tabular_encoder")
            os.makedirs(save_dir, exist_ok=True)
            rank0_print(f"Saving table encoder to {save_dir}")
            save_file(model_to_save.table_encoder.state_dict(), os.path.join(save_dir, "model.safetensors"))
            with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(model_to_save.table_encoder.config.to_dict(), f, ensure_ascii=False, indent=2)
            rank0_print(f"Saved table encoder only to {save_dir}")
    else:
        rank0_print(f"Saving full model to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
