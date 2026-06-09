import os
import time
import uuid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.TableEncoder.text_encoder import TextEncoder


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
    query_encoder: str,
    max_length: int,
    query_llm_model_path: str,
    knowledge_encoder_path: str,
    knowledge_encoder_base_model_path: str,
    batch_size: int = 16,
):
    if query_encoder == "llm":
        return build_query_embeddings(
            query_texts,
            cache_path,
            query_llm_model_path,
            max_length,
            batch_size=batch_size,
        )
    if query_encoder == "knowledge":
        return build_knowledge_query_embeddings(
            query_texts=query_texts,
            cache_path=cache_path,
            model_path=knowledge_encoder_path,
            base_model_path=knowledge_encoder_base_model_path,
            max_length=max_length,
            batch_size=batch_size,
        )
    raise ValueError(f"Unsupported query_encoder: {query_encoder}. Expected 'llm' or 'knowledge'.")
