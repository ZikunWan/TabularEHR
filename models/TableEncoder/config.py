from typing import Optional, List
from transformers import PretrainedConfig

class TableEncoderConfig(PretrainedConfig):
    """
    Configuration for the LongTableEncoder model.
    """
    model_type = "table_encoder"

    def __init__(
        self,
        # Core Dimension Settings
        text_dim: int = 768,
        dim: int = 768,
        depth: int = 6,
        heads: int = 12,
        dim_head: int = 64,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        
        # Q-Former / Output Adapter Settings
        num_queries: int = 24,
        dim_out: Optional[int] = None,
        
        # Feature Engineering Settings
        fourier_scales: List[float] = None,
        type_vocab_size: int = 11,
        
        # Attention Strategy: '1d', '2d_grid', or 'hierarchical'
        attention_mode: str = '1d',
        
        # Classifier specific
        num_classes: int = 2,
        num_points: Optional[int] = None,
        num_metrics: Optional[int] = None,
        **kwargs
    ):
        if fourier_scales is None:
            fourier_scales = [0.1, 1., 10., 100.]
            
        self.text_dim = text_dim
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        
        self.num_queries = num_queries
        self.dim_out = dim_out
        
        self.fourier_scales = fourier_scales
        self.type_vocab_size = type_vocab_size
        self.attention_mode = attention_mode
        
        self.num_classes = num_classes
        self.num_points = num_points
        self.num_metrics = num_metrics
        
        super().__init__(**kwargs)
