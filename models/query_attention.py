from typing import Optional

import torch
import torch.nn as nn

from models.TableEncoder.attention import CrossFlashAttention, RMSNorm
from models.TableEncoder.config import LongTableEncoder1DConfig


class QueryCrossAttentionHead(nn.Module):
    def __init__(self, config: LongTableEncoder1DConfig, query_dim: int):
        super().__init__()
        self.query_norm = RMSNorm(query_dim)
        self.context_norm = RMSNorm(query_dim)
        self.cross_attn = CrossFlashAttention(
            dim=query_dim,
            num_heads=query_dim // config.dim_head,
            attn_drop=config.dropout,
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, query_dim),
        )
        self.norm2 = nn.LayerNorm(query_dim)

    def forward(
        self,
        query_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        squeeze_query = query_embeds.dim() == 2
        query = self.query_norm(query_embeds.to(dtype=hidden_states.dtype))
        if squeeze_query:
            query = query.unsqueeze(1)
        hidden_states = self.context_norm(hidden_states)
        attn_out = self.cross_attn(query, hidden_states, seq_mask)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))
        if squeeze_query:
            query = query.squeeze(1)
        return query
