import torch

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
