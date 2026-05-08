import os
import time

import torch
from transformers import AutoTokenizer

from models.TableEncoder.text_encoder import TextEncoder


def local_rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def load_checkpoint_state_dict(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def load_query_text_encoder(model_path: str, base_model_path: str, device: torch.device):
    model_name = base_model_path if model_path.endswith(".pt") else model_path
    model = TextEncoder(model_name)
    if model_path.endswith(".pt"):
        state_dict = load_checkpoint_state_dict(model_path)
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


def wait_for_query_cache(cache_path: str, query_keys):
    while True:
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            cached_embeddings = cache["embeddings"]
            if all(query_key in cached_embeddings for query_key in query_keys):
                return cache
        time.sleep(2)


def build_query_embeddings(
    query_texts: dict[str, str],
    cache_path: str,
    model_path: str,
    base_model_path: str,
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
    tokenizer_path = base_model_path if model_path.endswith(".pt") else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_query_text_encoder(model_path, base_model_path, device)
    model.eval()

    tokens = tokenizer(
        [query_texts[query_key] for query_key in missing_keys],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        query_embeds = model.encode_text(tokens).cpu()

    for idx, query_key in enumerate(missing_keys):
        embeddings[query_key] = query_embeds[idx]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": model.hidden_size,
            "model_path": model_path,
            "base_model_path": base_model_path,
        },
        cache_path,
    )
    return {query_key: embeddings[query_key] for query_key in query_keys}, int(model.hidden_size)
