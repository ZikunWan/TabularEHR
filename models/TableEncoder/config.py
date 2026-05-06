from typing import List, Optional

from transformers import PretrainedConfig


class _BaseTableEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        # Core dimensions for input text embeddings and transformer hidden states.
        text_dim: int = 768,
        dim: int = 768,
        depth: int = 6,
        heads: int = 12,
        dim_head: int = 64,
        mlp_dim: int = 3072,
        dropout: float = 0.0,

        # Output adapter: allocate at most max_queries, about one query per tokens_per_query rows.
        max_queries: int = 160,
        tokens_per_query: int = 32,
        dim_out: Optional[int] = None,
        # Keep only the most recent max_table_len rows before encoding.
        max_table_len: Optional[int] = 32768,
        # GPT-style table encoding: each row can only attend to current/past rows.
        is_causal: bool = True,

        # Table feature settings for numeric value Fourier features and type embeddings.
        fourier_scales: Optional[List[float]] = None,
        type_vocab_size: int = 11,

        **kwargs,
    ):
        super().__init__(**kwargs)

        if fourier_scales is None:
            fourier_scales = [0.1, 1.0, 10.0, 100.0]
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        if tokens_per_query <= 0:
            raise ValueError("tokens_per_query must be positive")
        if max_table_len is not None and max_table_len <= 0:
            raise ValueError("max_table_len must be positive when set")

        self.text_dim = text_dim
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.max_queries = max_queries
        self.tokens_per_query = tokens_per_query
        self.dim_out = dim_out
        self.max_table_len = max_table_len
        self.is_causal = is_causal

        self.fourier_scales = fourier_scales
        self.type_vocab_size = type_vocab_size


class LongTableEncoder1DConfig(_BaseTableEncoderConfig):
    model_type = "long_table_encoder_1d"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LongTableEncoder2DConfig(_BaseTableEncoderConfig):
    model_type = "long_table_encoder_2d"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LongTableEncoderHierarchicalConfig(_BaseTableEncoderConfig):
    model_type = "long_table_encoder_hierarchical"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LongTableEncoderMemoryConfig(_BaseTableEncoderConfig):
    model_type = "long_table_encoder_memory"

    def __init__(
        self,
        # Historical rows are split into chunks before being summarized into memory tokens.
        memory_chunk_len: int = 512,
        # Recent rows are kept in full detail and appended after memory tokens.
        max_recent_len: int = 4096,
        # Keep at most this many memory tokens from historical chunks.
        max_memory_tokens: int = 64,
        # Chunk summary method: "mean", "cls", or "attention".
        memory_pooling: str = "cls",
        **kwargs,
    ):
        if memory_chunk_len <= 0:
            raise ValueError("memory_chunk_len must be positive")
        if max_recent_len <= 0:
            raise ValueError("max_recent_len must be positive")
        if max_memory_tokens <= 0:
            raise ValueError("max_memory_tokens must be positive")
        if memory_pooling not in {"mean", "cls", "attention"}:
            raise ValueError("memory_pooling must be one of: mean, cls, attention")

        super().__init__(**kwargs)
        self.memory_chunk_len = memory_chunk_len
        self.max_recent_len = max_recent_len
        self.max_memory_tokens = max_memory_tokens
        self.memory_pooling = memory_pooling
