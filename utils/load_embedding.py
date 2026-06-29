import os
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.TableEncoder.text_encoder import TextEncoder

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, MASK_TOKEN]

_embedding_cache = None
_text_dim = None


def load_embedding_cache(cache_path: str):
    """
    Load pre-computed embedding cache for the given dataset.
    """
    global _embedding_cache, _text_dim
    
    if _embedding_cache is None:
        data = torch.load(cache_path, map_location='cpu', weights_only=False)
        _embedding_cache = data['embeddings']
        _text_dim = data['text_dim']
        print(f"Loaded {len(_embedding_cache)} embeddings (dim={_text_dim}) from {cache_path}")

    return _embedding_cache, _text_dim


def build_vocab_keys(embedding_cache: dict | None = None) -> list[str]:
    """
    Build table text vocab keys from cache keys plus special tokens.
    """
    if embedding_cache is None:
        embedding_cache = _embedding_cache

    vocab_keys = list(SPECIAL_TOKENS)
    vocab_keys.extend([text for text in embedding_cache.keys() if text not in SPECIAL_TOKENS])
    return vocab_keys


def build_text_to_idx(vocab_keys: list[str]) -> dict[str, int]:
    """
    Build text-to-index mapping for table text tokens.
    """
    return {text: idx for idx, text in enumerate(vocab_keys)}


def build_embedding_matrix(
    embedding_cache: dict | None = None,
    vocab_keys: list[str] | None = None,
) -> torch.Tensor:
    """
    Build embedding matrix aligned with vocab_keys.

    Cache strings use pre-computed embeddings. [PAD] is zero-initialized.
    [UNK] and [MASK] are randomly initialized and can be learned downstream.
    """
    if embedding_cache is None:
        embedding_cache = _embedding_cache
    if vocab_keys is None:
        vocab_keys = build_vocab_keys(embedding_cache)

    matrix = torch.empty(len(vocab_keys), _text_dim)
    torch.nn.init.normal_(matrix, mean=0.0, std=0.02)

    for idx, text in enumerate(vocab_keys):
        if text == PAD_TOKEN:
            matrix[idx].zero_()
        elif text in embedding_cache:
            matrix[idx] = embedding_cache[text]

    return matrix


def get_special_token_indices(text_to_idx: dict[str, int]) -> dict[str, int]:
    """
    Return ids for special table text tokens.
    """
    return {
        "pad_idx": text_to_idx[PAD_TOKEN],
        "unk_idx": text_to_idx[UNK_TOKEN],
        "mask_idx": text_to_idx[MASK_TOKEN],
    }


def get_embedding(text: str) -> torch.Tensor | None:
    """
    Look up the embedding for a given text literal.
    [PAD] is synthesized as a zero vector because raw embedding caches do not
    store special tokens. Returns None for other strings missing from the cache.
    """
    global _embedding_cache, _text_dim

    if _embedding_cache is None:
        raise RuntimeError("Embedding cache not loaded. Please call `load_embedding_cache(path)` first.")

    if text == PAD_TOKEN:
        return torch.zeros(_text_dim)

    return _embedding_cache.get(text)


def get_pad_embedding() -> torch.Tensor:
    return get_embedding(PAD_TOKEN)


def local_rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def wait_for_query_cache(cache_path: str, query_keys):
    while True:
        if os.path.exists(cache_path):
            try:
                cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                cached_embeddings = cache["embeddings"]
                if all(query_key in cached_embeddings for query_key in query_keys):
                    return cache
            except (EOFError, RuntimeError, OSError):
                pass
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
    batch_size: int = 16,
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

    text_dim = None
    for start in range(0, len(missing_keys), batch_size):
        end = min(start + batch_size, len(missing_keys))
        batch_texts = missing_texts[start:end]
        tokens = tokenizer(
            batch_texts,
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
        text_dim = int(query_embeds.size(-1))
        for idx, query_key in enumerate(missing_keys[start:end]):
            embeddings[query_key] = query_embeds[idx]
        del tokens, outputs, last_hidden, query_embeds
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if text_dim is None:
        text_dim = int(next(iter(embeddings.values())).numel())
    tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": text_dim,
            "model_path": model_path,
        },
        tmp_cache_path,
    )
    os.replace(tmp_cache_path, cache_path)
    return {query_key: embeddings[query_key] for query_key in query_keys}, text_dim


def _load_knowledge_encoder(model_path: str, base_model_path: str, device: torch.device):
    checkpoint_path = model_path
    if os.path.isdir(model_path):
        checkpoint_path = next(
            (
                os.path.join(model_path, filename)
                for filename in ("model.safetensors", "pytorch_model.bin", "best.pt")
                if os.path.exists(os.path.join(model_path, filename))
            ),
            None,
        )

    tokenizer_path = model_path
    if os.path.isfile(model_path):
        tokenizer_path = os.path.dirname(model_path)
    if not os.path.exists(os.path.join(tokenizer_path, "tokenizer_config.json")):
        tokenizer_path = base_model_path

    model = TextEncoder(base_model_path)
    if checkpoint_path:
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        model_state = model.state_dict()
        matched_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and value.shape == model_state[key].shape
        }
        model.load_state_dict(matched_state, strict=False)

    return model.to(device).eval(), tokenizer_path


def build_knowledge_query_embeddings(
    query_texts: dict[str, str],
    cache_path: str,
    model_path: str,
    base_model_path: str,
    max_length: int,
    batch_size: int = 16,
):
    query_texts = {str(key): str(text) for key, text in query_texts.items()}
    query_keys = sorted(query_texts.keys())

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("model_path") != model_path:
            raise ValueError(
                f"Query cache encoder mismatch: {cache_path} was created with "
                f"{cache.get('model_path')!r}, requested {model_path!r}. Use a separate cache path."
            )
        cached_embeddings = cache["embeddings"]
        if all(query_key in cached_embeddings for query_key in query_keys):
            embeddings = {query_key: cached_embeddings[query_key].float() for query_key in query_keys}
            return embeddings, int(cache["text_dim"])

    if not local_rank0():
        cache = wait_for_query_cache(cache_path, query_keys)
        cached_embeddings = cache["embeddings"]
        embeddings = {query_key: cached_embeddings[query_key].float() for query_key in query_keys}
        return embeddings, int(cache["text_dim"])

    embeddings = {}
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        embeddings.update({key: value.float() for key, value in cache["embeddings"].items()})

    missing_keys = [query_key for query_key in query_keys if query_key not in embeddings]
    if missing_keys:
        device = torch.device(
            f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu"
        )
        model, tokenizer_path = _load_knowledge_encoder(model_path, base_model_path, device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        missing_texts = [query_texts[query_key] for query_key in missing_keys]

        for start in range(0, len(missing_keys), batch_size):
            end = min(start + batch_size, len(missing_keys))
            tokens = tokenizer(
                missing_texts[start:end],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.no_grad():
                query_embeds = model.encode_text(tokens).float().cpu()
            for offset, query_key in enumerate(missing_keys[start:end]):
                embeddings[query_key] = query_embeds[offset]

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    text_dim = int(next(iter(embeddings.values())).numel())
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": text_dim,
            "model_path": model_path,
            "base_model_path": base_model_path,
        },
        tmp_cache_path,
    )
    os.replace(tmp_cache_path, cache_path)
    return {query_key: embeddings[query_key] for query_key in query_keys}, text_dim


def build_task_query_embeddings(
    query_texts: dict[str, str],
    cache_path: str,
    max_length: int,
    knowledge_encoder_path: str,
    knowledge_encoder_base_model_path: str,
    batch_size: int = 16,
):
    return build_knowledge_query_embeddings(
        query_texts=query_texts,
        cache_path=cache_path,
        model_path=knowledge_encoder_path,
        base_model_path=knowledge_encoder_base_model_path,
        max_length=max_length,
        batch_size=batch_size,
    )
   
