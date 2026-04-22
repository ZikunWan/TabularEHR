import json
import os
import random
import sys
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.mimic.mimic_dataset import MIMICIV
from models.TableEncoder.config import TableEncoderConfig
from models.TableEncoder.encoder import LongTableEncoder
from pretraining.bi_reconstruct import (
    TABLE_PLACEHOLDER_TOKEN,
    ModelArguments as TrainModelArguments,
    _prepare_random_subset_csv,
    rank0_print,
)
from utils.collate import build_table_token_tensors
from utils.load_embedding import load_embedding_cache


@dataclass
class ModelArguments:
    llm_path: str = field(default="/data/model_weights_public/BlueZeros/EHR-R1-1.7B")
    table_encoder_path: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/stage2_bi_reconstruct/tabular_encoder"
    )
    attention_mode: str = field(default="1d")
    projector_hidden_size: int = field(default=2048)
    bf16: bool = field(default=True)


@dataclass
class DataArguments:
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    test_sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/test/bi_reconstruct.csv"
    )
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    table_text_embedding: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/embeddings/table_text_embeddings.pt"
    )
    random_subset_size: Optional[int] = field(default=None)
    random_subset_seed: int = field(default=42)
    max_samples: Optional[int] = field(default=20)
    lazy_mode: bool = field(default=True)
    mask_ratio: float = field(default=0.15)
    max_masked_cells: int = field(default=64)
    output_path: Optional[str] = field(default=None)
    seed: int = field(default=42)


@dataclass
class GenerationArguments:
    task: str = field(default="text_to_table")
    max_new_tokens: int = field(default=256)
    print_samples: int = field(default=5)


def _normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            if key.startswith("module."):
                key = key[len("module.") :]
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod.") :]
        normalized[key] = value
    return normalized


def _resolve_encoder_files(table_encoder_path: str) -> Tuple[str, Optional[str]]:
    if os.path.isdir(table_encoder_path):
        state_candidates = [
            os.path.join(table_encoder_path, "model.safetensors"),
            os.path.join(table_encoder_path, "pytorch_model.bin"),
        ]
        config_path = os.path.join(table_encoder_path, "config.json")
    else:
        state_candidates = [table_encoder_path]
        config_path = os.path.join(os.path.dirname(table_encoder_path), "config.json")

    state_path = None
    for candidate in state_candidates:
        if os.path.isfile(candidate):
            state_path = candidate
            break

    if state_path is None:
        raise FileNotFoundError(
            f"Cannot find table encoder weights under: {table_encoder_path}. "
            "Expected `model.safetensors` or `pytorch_model.bin`."
        )

    if not os.path.isfile(config_path):
        config_path = None

    return state_path, config_path


def _build_encoder_config(
    config_path: Optional[str],
    text_dim: int,
    type_vocab_size: int,
    model_args: ModelArguments,
) -> TableEncoderConfig:
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
    else:
        cfg_dict = {}

    cfg_dict["text_dim"] = text_dim
    cfg_dict["type_vocab_size"] = type_vocab_size
    cfg_dict.setdefault("attention_mode", model_args.attention_mode)
    cfg_dict.setdefault("dim_out", model_args.projector_hidden_size)

    return TableEncoderConfig(**cfg_dict)


def _load_table_encoder(
    model_args: ModelArguments,
    text_dim: int,
    type_vocab_size: int,
    device: torch.device,
) -> LongTableEncoder:
    state_path, config_path = _resolve_encoder_files(model_args.table_encoder_path)
    config = _build_encoder_config(config_path, text_dim, type_vocab_size, model_args)
    encoder = LongTableEncoder(config=config)

    if state_path.endswith(".safetensors"):
        state_dict = load_file(state_path, device="cpu")
    else:
        state_dict = torch.load(state_path, map_location="cpu")
    state_dict = _normalize_state_dict_keys(state_dict)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.to(device)
    encoder.eval()

    rank0_print(f"Loaded table encoder from {state_path}")
    return encoder


def _mask_table(df, sample_uid: str, mask_ratio: float, max_masked_cells: int, seed: int):
    work_df = df.reset_index(drop=True).copy()

    candidates = []
    for row_id in range(len(work_df)):
        item = str(work_df.at[row_id, "Item"]).strip()
        value = str(work_df.at[row_id, "Value"]).strip()
        if value and value.lower() != "nan":
            candidates.append((row_id, "Value"))
        elif item and item.lower() != "nan":
            candidates.append((row_id, "Item"))

    if not candidates:
        return work_df, "[]"

    stable_hash = int(hashlib.md5(sample_uid.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random((stable_hash ^ seed) & 0xFFFFFFFF)
    num_mask = int(len(candidates) * mask_ratio)
    num_mask = max(1, min(num_mask, max_masked_cells, len(candidates)))
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


def _build_text_to_table_prompt(markdown_text: str) -> str:
    return (
        "Reconstruct the table cells masked in the table embedding by using the full markdown content as reference.\n"
        "Markdown Content is fully observed and does not contain [EMPTY].\n"
        "The masked positions exist only in the table embedding.\n"
        "Output JSON list only.\n"
        "Each element: {\"Item\": str, \"Value\": str}\n"
        "\n"
        "[Table Embedding]\n"
        f"{TABLE_PLACEHOLDER_TOKEN}\n\n"
        "Markdown Content:\n"
        f"{markdown_text}\n\n"
        "Output:\n"
    )


def _build_prompt_inputs(llm, tokenizer, prompt_text: str, table_embeds: torch.Tensor, placeholder_token_id: int):
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(table_embeds.device)
    placeholder_pos = (prompt_ids[0] == placeholder_token_id).nonzero(as_tuple=False)
    if placeholder_pos.numel() == 0:
        raise ValueError("Prompt does not contain table placeholder token.")
    placeholder_idx = int(placeholder_pos[0].item())

    embed_layer = llm.get_input_embeddings()
    model_dtype = embed_layer.weight.dtype
    table_embeds = table_embeds.to(dtype=model_dtype)

    embed_parts = []
    if placeholder_idx > 0:
        embed_parts.append(embed_layer(prompt_ids[:, :placeholder_idx]))
    embed_parts.append(table_embeds)
    if placeholder_idx + 1 < prompt_ids.size(1):
        embed_parts.append(embed_layer(prompt_ids[:, placeholder_idx + 1 :]))

    inputs_embeds = torch.cat(embed_parts, dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=table_embeds.device, dtype=torch.long)
    return inputs_embeds, attention_mask


@torch.inference_mode()
def _greedy_generate_with_table(
    llm,
    tokenizer,
    table_embeds: torch.Tensor,
    prompt_text: str,
    placeholder_token_id: int,
    max_new_tokens: int,
) -> str:
    inputs_embeds, attention_mask = _build_prompt_inputs(
        llm=llm,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        table_embeds=table_embeds,
        placeholder_token_id=placeholder_token_id,
    )

    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids = []
    prompt_len = attention_mask.size(1)
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if eos_token_id is not None and token_id == eos_token_id:
            break

        generated_ids.append(token_id)
        step_attention_mask = torch.ones(
            (1, prompt_len + len(generated_ids)),
            device=attention_mask.device,
            dtype=torch.long,
        )
        outputs = llm(
            input_ids=next_token,
            attention_mask=step_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _encode_table(table_encoder, table_embedding, table_batch, device: torch.device):
    return table_encoder(
        item_emb=table_embedding(table_batch["item_ids"].to(device)),
        unit_emb=table_embedding(table_batch["unit_ids"].to(device)),
        value_emb=table_embedding(table_batch["value_text_ids"].to(device)),
        times=table_batch["times"].to(device),
        numeric_values=table_batch["numeric_values"].to(device),
        numeric_mask=table_batch["numeric_mask"].to(device),
        type_ids=table_batch["type_ids"].to(device),
        seq_mask=table_batch["seq_mask"].to(device),
    )


def _default_output_path(table_encoder_path: str, task: str) -> str:
    base_dir = table_encoder_path if os.path.isdir(table_encoder_path) else os.path.dirname(table_encoder_path)
    return os.path.join(base_dir, f"bi_reconstruct_{task}_predictions.jsonl")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments))
    model_args, data_args, gen_args = parser.parse_args_into_dataclasses()

    if gen_args.task not in {"text_to_table", "table_to_text", "both"}:
        raise ValueError("--task must be one of: text_to_table, table_to_text, both")

    set_seed(data_args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank0_print(f"Using device: {device}")

    test_sample_info_path = _prepare_random_subset_csv(
        data_args.test_sample_info_path,
        data_args.random_subset_size,
        data_args.random_subset_seed,
    )
    rank0_print(f"Test sample info: {test_sample_info_path}")

    with open(data_args.type_vocab_file, "r", encoding="utf-8") as f:
        type_vocab = json.load(f)

    embedding_map, text_dim = load_embedding_cache(data_args.table_text_embedding)
    vocab_keys = list(embedding_map.keys())
    text_to_idx = {text: idx for idx, text in enumerate(vocab_keys)}
    embedding_matrix = torch.stack([embedding_map[key] for key in vocab_keys]).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_args.llm_path, trust_remote_code=True, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [TABLE_PLACEHOLDER_TOKEN]})
    table_placeholder_token_id = tokenizer.convert_tokens_to_ids(TABLE_PLACEHOLDER_TOKEN)

    llm_dtype = torch.bfloat16 if model_args.bf16 and device.type == "cuda" else torch.float32
    llm = AutoModelForCausalLM.from_pretrained(
        model_args.llm_path,
        trust_remote_code=True,
        dtype=llm_dtype,
    )
    llm.resize_token_embeddings(len(tokenizer))
    llm.to(device)
    llm.eval()

    table_encoder = _load_table_encoder(
        model_args=model_args,
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        device=device,
    )
    table_token_embedding = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(device)
    table_token_embedding.eval()

    dataset = MIMICIV(
        root_dir=data_args.root_dir,
        sample_info_path=test_sample_info_path,
        lazy_mode=data_args.lazy_mode,
        shuffle=False,
        table_mode="table_only",
        max_samples=data_args.max_samples,
        use_table_length_cache=True,
    )

    output_path = data_args.output_path or _default_output_path(model_args.table_encoder_path, gen_args.task)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    rank0_print(f"Writing predictions to: {output_path}")
    rank0_print(f"Eval samples: {len(dataset)}")

    prompt_defaults = TrainModelArguments()

    with open(output_path, "w", encoding="utf-8") as writer:
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_meta = dataset.sample_info[idx] if hasattr(dataset, "sample_info") else {}
            sample_uid = str(sample.get("idx", idx))
            table_df = sample["measurement_table"]
            prompt_text = str(sample.get("input", ""))

            record = {
                "idx": sample.get("idx", idx),
                "subject_id": sample_meta.get("subject_id", ""),
                "task": sample_meta.get("task", sample.get("task_info", {}).get("name", "")),
            }

            if gen_args.task in {"text_to_table", "both"}:
                masked_df, target_json = _mask_table(
                    df=table_df,
                    sample_uid=sample_uid,
                    mask_ratio=data_args.mask_ratio,
                    max_masked_cells=data_args.max_masked_cells,
                    seed=data_args.seed,
                )
                masked_batch = build_table_token_tensors(
                    tables_list=[masked_df],
                    text_to_idx=text_to_idx,
                    pad_idx=text_to_idx.get("[PAD]", 0),
                    type_vocab=type_vocab,
                )
                masked_table_embeds = _encode_table(table_encoder, table_token_embedding, masked_batch, device=device)
                text_to_table_pred = _greedy_generate_with_table(
                    llm=llm,
                    tokenizer=tokenizer,
                    table_embeds=masked_table_embeds,
                    prompt_text=_build_text_to_table_prompt(prompt_text),
                    placeholder_token_id=table_placeholder_token_id,
                    max_new_tokens=gen_args.max_new_tokens,
                )
                record["text_to_table"] = {
                    "prompt_markdown": prompt_text,
                    "target": target_json,
                    "prediction": text_to_table_pred,
                }

            if gen_args.task in {"table_to_text", "both"}:
                full_batch = build_table_token_tensors(
                    tables_list=[table_df],
                    text_to_idx=text_to_idx,
                    pad_idx=text_to_idx.get("[PAD]", 0),
                    type_vocab=type_vocab,
                )
                full_table_embeds = _encode_table(table_encoder, table_token_embedding, full_batch, device=device)
                table_to_text_pred = _greedy_generate_with_table(
                    llm=llm,
                    tokenizer=tokenizer,
                    table_embeds=full_table_embeds,
                    prompt_text=prompt_defaults.table_to_text_prompt,
                    placeholder_token_id=table_placeholder_token_id,
                    max_new_tokens=gen_args.max_new_tokens,
                )
                record["table_to_text"] = {
                    "target": prompt_text,
                    "prediction": table_to_text_pred,
                }

            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            if idx < gen_args.print_samples:
                rank0_print("=" * 80)
                rank0_print(f"Sample {idx} / idx={record['idx']}")
                if "text_to_table" in record:
                    rank0_print("[text_to_table] target:")
                    rank0_print(record["text_to_table"]["target"])
                    rank0_print("[text_to_table] prediction:")
                    rank0_print(record["text_to_table"]["prediction"])
                if "table_to_text" in record:
                    rank0_print("[table_to_text] target snippet:")
                    rank0_print(record["table_to_text"]["target"][:400])
                    rank0_print("[table_to_text] prediction snippet:")
                    rank0_print(record["table_to_text"]["prediction"][:400])

    rank0_print("Done.")


if __name__ == "__main__":
    main()
