import torch
import torch.nn as nn
import math
from typing import Optional

class TimeEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer('div_term', torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)))

    def forward(self, t):
        """
        t: [batch_size, seq_len] - absolute time or relative time
        """
        t = t.unsqueeze(-1) # [batch_size, seq_len, 1]
        div_term = self.div_term.to(t.device)
        
        pe = torch.zeros(*t.shape[:2], self.dim, device=t.device, dtype=t.dtype)
        pe[..., 0::2] = torch.sin(t * div_term)
        pe[..., 1::2] = torch.cos(t * div_term)
        return pe


class FourierFeatures(nn.Module):
    def __init__(self, fourier_scales=[0.1, 1., 10., 100.]):
        super(FourierFeatures, self).__init__()
        assert len(fourier_scales) > 0

        _omega = 1 / (2 ** torch.tensor(fourier_scales, dtype=torch.float32))
        self.register_buffer('omega', _omega)

        self.output_dim = 2 * len(self.omega)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = torch.einsum('bs,d -> bsd', x, self.omega.to(dtype=x.dtype))
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        res = torch.cat((x_sin, x_cos), -1)
        return res


class LongTableEmbedding(nn.Module):
    """
    Handles fusion of pre-computed Item, Value, Unit embeddings with Time encoding.
    """
    def __init__(self, text_dim: int = 768,
                 dim: int = 768,
                 type_vocab_size: int = 24,
                 fourier_scales=[0.1, 1., 10., 100.]):
        """
        Args:
            text_dim: Dimension of pre-computed text embeddings
            dim: Model hidden dimension
            type_vocab_size: vocab size for type category (e.g. Lab, Vital, Med)
            fourier_scales: scales for numeric fourier features
        """
        super().__init__()
        self.text_dim = text_dim
        self.dim = dim
        
        # Embedding Projection Layers
        self.item_proj = nn.Linear(text_dim, dim)
        self.unit_proj = nn.Linear(text_dim, dim)
        self.value_text_proj = nn.Linear(text_dim, dim)
        
        # Type Category Embedding
        self.type_embedding = nn.Embedding(type_vocab_size, dim)
        
        # Numeric Value Fourier Features + Projection
        self.fourier_feat = FourierFeatures(fourier_scales)
        self.numeric_proj = nn.Linear(self.fourier_feat.output_dim, dim)
        
        # Time Encoding
        self.time_enc = TimeEncoding(dim)

    def forward(self, 
                item_emb: torch.Tensor,
                unit_emb: torch.Tensor,
                value_emb: torch.Tensor,
                times: torch.Tensor,
                numeric_values: torch.Tensor,
                numeric_mask: torch.Tensor,
                type_ids: Optional[torch.Tensor] = None,
                use_2d_attention: bool = False):
        """
        Forward pass with pre-computed embeddings.
        
        Args:
            item_emb: [batch_size, seq_len, text_dim] - Pre-computed item embeddings
            unit_emb: [batch_size, seq_len, text_dim] - Pre-computed unit embeddings
            value_emb: [batch_size, seq_len, text_dim] - Pre-computed value text embeddings
            times: [batch_size, seq_len] - Time values
            numeric_values: [batch_size, seq_len] - Numeric values (0 for non-numeric)
            numeric_mask: [batch_size, seq_len] - 1 for numeric, 0 for text
            type_ids: [batch_size, seq_len] - Type Category IDs (Optional)
            
        Returns:
            embeddings: [batch_size, seq_len, dim]
        """
        batch_size, seq_len = times.shape
        
        # 1. Project Item embeddings
        item_proj = self.item_proj(item_emb)  # [batch_size, seq_len, dim]
        
        # 2. Project Unit embeddings
        unit_proj = self.unit_proj(unit_emb)  # [batch_size, seq_len, dim]
        
        # 3. Handle Values (hybrid numeric/text)
        # Text path
        val_text_proj = self.value_text_proj(value_emb)  # [batch_size, seq_len, dim]
        
        # Numeric path (Fourier Features -> Linear Projection)
        val_numeric = self.fourier_feat(numeric_values) # [batch_size, seq_len, fourier_dim]
        val_numeric = self.numeric_proj(val_numeric) # [batch_size, seq_len, dim]
        
        # Combine: use numeric where mask=1, else use text
        numeric_mask_expanded = numeric_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        val_proj = numeric_mask_expanded * val_numeric + (1 - numeric_mask_expanded) * val_text_proj
        
        # 4. Fusion
        event_emb = item_proj + val_proj + unit_proj  # [batch_size, seq_len, dim]
        
        # Add Type Embedding if provided
        type_emb = None
        if type_ids is not None:
            type_emb = self.type_embedding(type_ids)
            event_emb = event_emb + type_emb
        
        # 5. Add Time Encoding 
        time_emb =  self.time_enc(times)
        time_mask = (times != 0).unsqueeze(-1).to(time_emb.dtype)
        
        if use_2d_attention:
            return item_proj, val_proj, unit_proj, type_emb, time_emb, time_mask
        else:
            x = event_emb + time_emb * time_mask
            return x
