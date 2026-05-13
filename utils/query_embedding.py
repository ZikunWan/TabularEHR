import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def local_rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def wait_for_query_cache(cache_path: str, query_keys):
    while True:
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_embeddings = cache["embeddings"]
            if all(query_key in cached_embeddings for query_key in query_keys):
                return cache
        time.sleep(2)


def get_llm_pooling_indices(model_path: str, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    model_name_lower = model_path.lower()
    eos_id = None
    if "qwen" in model_name_lower or "ehr-r1" in model_name_lower:
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    if eos_id is not None:
        eos_mask = input_ids == eos_id
        if eos_mask.sum().item() > 0:
            seq_len = input_ids.size(1)
            indices = torch.arange(seq_len, device=input_ids.device)
            return (eos_mask * indices).argmax(dim=1)

    return attention_mask.sum(dim=1) - 1


def build_query_embeddings(
    query_texts: dict[str, str],
    cache_path: str,
    model_path: str,
    max_length: int,
):
    query_texts = {str(key): str(text) for key, text in query_texts.items()}
    query_keys = sorted(query_texts.keys())

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_embeddings = cache["embeddings"]
        if all(query_key in cached_embeddings for query_key in query_keys):
            return {query_key: cached_embeddings[query_key] for query_key in query_keys}, int(cache["text_dim"])

    if not local_rank0():
        cache = wait_for_query_cache(cache_path, query_keys)
        cached_embeddings = cache["embeddings"]
        return {query_key: cached_embeddings[query_key] for query_key in query_keys}, int(cache["text_dim"])

    embeddings = {}
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        embeddings.update(cache["embeddings"])

    missing_keys = [query_key for query_key in query_keys if query_key not in embeddings]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": device},
    ).eval()

    missing_texts = [query_texts[query_key] for query_key in missing_keys]
    if tokenizer.chat_template:
        missing_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": query_text}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for query_text in missing_texts
        ]

    tokens = tokenizer(
        missing_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        last_indices = get_llm_pooling_indices(
            model_path,
            tokenizer,
            tokens["input_ids"],
            tokens["attention_mask"],
        )
        query_embeds = last_hidden[
            torch.arange(last_hidden.size(0), device=last_hidden.device),
            last_indices,
        ].cpu().to(torch.bfloat16)

    for idx, query_key in enumerate(missing_keys):
        embeddings[query_key] = query_embeds[idx]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": int(query_embeds.size(-1)),
            "model_path": model_path,
        },
        cache_path,
    )
    return {query_key: embeddings[query_key] for query_key in query_keys}, int(query_embeds.size(-1))
