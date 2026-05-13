import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import trunc_normal_

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
    
def get_1d_sincos_pos_embed_from_grid(embed_dim, length):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (num_positions,)
    out: (num_positions, embed_dim)
    """
    positions = np.arange(length)
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (embed_dim/2,)

    outer_product = np.einsum('m,d->md', positions, omega)  # (num_positions, embed_dim/2), outer product

    embedding_sin = np.sin(outer_product) # (num_positions, embed_dim/2)
    embedding_cos = np.cos(outer_product) # (num_positions, embed_dim/2)

    positional_embedding = np.concatenate([embedding_sin, embedding_cos], axis=1)  # (num_positions, embed_dim)
    positional_embedding = torch.from_numpy(positional_embedding).float()
    return positional_embedding

class QFormer(nn.Module):
    def __init__(self, dim,
                    max_queries,
                    tokens_per_query=32,
                    dim_head=64,
                    mlp_type='identity',
                    dim_out=None):
        super().__init__()
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        if tokens_per_query <= 0:
            raise ValueError("tokens_per_query must be positive")
        self.max_queries = max_queries
        self.tokens_per_query = tokens_per_query
        self.dim_head = dim_head
        self.encoder_hidden_size = dim
        self.decoder_hidden_size = dim

        self.query = nn.Parameter(torch.randn(max_queries, self.decoder_hidden_size))
        trunc_normal_(self.query, std=.02)
        
        self.kv_proj = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size, bias=False)
        self.attn = nn.MultiheadAttention(self.decoder_hidden_size, num_heads = self.decoder_hidden_size // dim_head, batch_first=True)
        self.ln_q = nn.LayerNorm(self.decoder_hidden_size)
        self.ln_kv = nn.LayerNorm(self.decoder_hidden_size)
        
        if mlp_type == 'identity':
            self.mlp = Identity()
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size),
                nn.GELU(),
                nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size)
            )
            
        if dim_out:
             self.out_proj = nn.Linear(self.decoder_hidden_size, dim_out)
        else:
             self.out_proj = nn.Identity()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        from torch.nn.init import trunc_normal_
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, context, context_mask=None):
        # context: (batch_size, seq_len, encoder_hidden_size)
        # Ensure input dtype matches model weights (e.g. when using bf16 via DeepSpeed)
        context = context.to(dtype=self.kv_proj.weight.dtype)
        projected_context = self.kv_proj(context)
        projected_context = self.ln_kv(projected_context)
        
        key_tensor = projected_context
        value_tensor = projected_context

        batch_size, seq_len = projected_context.shape[:2]
        if context_mask is not None:
            valid_len = context_mask.long().sum(dim=1).max().item()
        else:
            valid_len = seq_len
        num_queries = min(
            self.max_queries,
            max(1, (int(valid_len) + self.tokens_per_query - 1) // self.tokens_per_query),
        )

        query_tensor = self.query[:num_queries].unsqueeze(0).repeat(batch_size, 1, 1)
        
        query_pos_embeds = get_1d_sincos_pos_embed_from_grid(self.decoder_hidden_size, num_queries)
        query_pos_embeds = query_pos_embeds.to(dtype=query_tensor.dtype, device=query_tensor.device)
        query_pos_embeds.requires_grad_(False)
        
        query_tensor = self.ln_q(query_tensor) + query_pos_embeds
        
        # context_mask: (batch_size, seq_len). 1 is valid, 0 is pad
        # MultiheadAttention requires key_padding_mask where True = ignore padding
        key_padding_mask = (context_mask == 0) if context_mask is not None else None
        
        attention_output = self.attn(query_tensor, key_tensor, value_tensor, key_padding_mask)[0]
        mlp_output = self.mlp(attention_output)
        return self.out_proj(mlp_output)


class QFormerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qformer = QFormer(
            config.dim,
            config.max_queries,
            tokens_per_query=config.tokens_per_query,
            dim_head=config.dim_head,
            dim_out=config.dim_out,
        )

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        return self.qformer(hidden_states, attention_mask)
