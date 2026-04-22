import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._attn = None
        self._attn_gradients = None
        self.attn_drop_prob = attn_drop

    def forward(self, x, mask=None):
        """
        :param x: Shape (batch_size, seq_len, hidden_dim)
        :param mask: 0 = ignored, 1 = attention enabled
        """
        self._attn = None
        self._attn_gradients = None

        batch_size, seq_len, hidden_dim = x.shape

        # [batch_size, seq_len, 3, num_heads, head_dim] -> [3, batch_size, num_heads, seq_len, head_dim]
        qkv = self.qkv(x).contiguous().reshape(batch_size, seq_len, 3, self.num_heads, hidden_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Handle mask for SDPA
        # Use float mask (-inf for masked positions, 0.0 for valid) instead of bool,
        # which is compatible with all SDPA backends (flash, mem_efficient, math).
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask.contiguous().view(batch_size, 1, 1, seq_len)  # [B, 1, 1, N]
            elif mask.dim() == 3:
                attn_mask = mask.contiguous().view(batch_size, 1, seq_len, seq_len)  # [B, 1, N, N]
            else:
                attn_mask = mask
            # Use a bool mask (True = attend, False = ignore).
            # SDPA accepts bool masks and can still select Flash / mem_efficient backends,
            # whereas a float additive mask (with -inf) forces the O(N²) Math backend.
            attn_mask = attn_mask.bool()

        # Flash Attention has a known backward bug for seq_len < 64 (e.g. intra-component
        # attention where seq_len == num_components == 4).  Use mem_efficient backend which:
        #   - avoids Flash Attention's small-seq backward CUDA bug
        #   - uses O(N) memory (unlike math backend which is O(N²) and causes OOM in hierarchical)
        if seq_len < 64:
            ctx = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
            
            with ctx:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop_prob if self.training else 0.0,
                    is_causal=False
                )
        else:
            # seq_len >= 64: let SDPA auto-select Flash Attention (O(N) memory)
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,  # bool mask → Flash Attention can be selected
                dropout_p=self.attn_drop_prob if self.training else 0.0,
                is_causal=False
            )

        x = x.transpose(1, 2).contiguous().reshape(batch_size, seq_len, hidden_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _save_attn_grad(self, grad):
        self._attn_gradients = grad

    def get_attn(self):
        return self._attn

    def get_attn_grad(self):
        return self._attn_gradients


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** 0.5
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # SwiGLU needs double the hidden dimension for the chunking
        inner_dim = int(hidden_dim * 2 / 3) 
        self.w12 = nn.Linear(dim, inner_dim * 2, bias=False)
        self.swiglu = SwiGLU()
        self.dropout1 = nn.Dropout(dropout)
        self.w3 = nn.Linear(inner_dim, dim, bias=False)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.w12(x)
        x = self.swiglu(x)
        x = self.dropout1(x)
        x = self.w3(x)
        x = self.dropout2(x)
        return x

class StandardTransformerLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FlashAttention(dim, num_heads=heads, attn_drop=dropout, proj_drop=dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x.to(self.norm1.weight.dtype)), mask)
        x = x + self.ffn(self.norm2(x.to(self.norm2.weight.dtype)))
        return x


class ComponentTransformerLayer(nn.Module):
    """
    A Transformer layer that computes 2D attention for tabular components:
    1. Intra-Measurement Attention: Attends across the components (e.g., Item, Value, Unit) of a single measurement.
    2. Inter-Measurement Attention: Attends across the sequence of measurements (time steps) for each component.
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        # Intra-Measurement Attention (Over components)
        self.norm1_intra = RMSNorm(dim)
        self.intra_attn = FlashAttention(dim, num_heads=heads, attn_drop=dropout, proj_drop=dropout)
        
        # Inter-Measurement Attention (Over sequence length)
        self.norm1_inter = RMSNorm(dim)
        self.inter_attn = FlashAttention(dim, num_heads=heads, attn_drop=dropout, proj_drop=dropout)
        
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, seq_mask=None):
        """
        x: [batch_size, seq_len, num_components, embed_dim]
        seq_mask: [batch_size, seq_len] (1 for valid, 0 for padding)
        """
        batch_size, seq_len, num_components, embed_dim = x.shape
        
        # 1. Intra-Measurement Attention (Across the components)
        # Reshape to [batch_size * seq_len, num_components, embed_dim]
        x_intra = x.reshape(batch_size * seq_len, num_components, embed_dim)
        
        # Apply intra attention. Mask is None because all components within a measurement are always valid
        norm_intra = self.norm1_intra(x_intra.to(self.norm1_intra.weight.dtype))
        out_intra = self.intra_attn(norm_intra, mask=None)
        x_intra = x_intra + out_intra
        
        # Reshape back
        x = x_intra.reshape(batch_size, seq_len, num_components, embed_dim)
        
        # 2. Inter-Measurement Attention (Across the sequence length)
        # Transpose to [batch_size, num_components, seq_len, embed_dim] -> [batch_size * num_components, seq_len, embed_dim]
        x_inter = x.transpose(1, 2).contiguous().reshape(batch_size * num_components, seq_len, embed_dim)
        
        # Expand seq_mask [batch_size, seq_len] for all components: [batch_size * num_components, seq_len]
        r_mask = None
        if seq_mask is not None:
            r_mask = (
                seq_mask.unsqueeze(1)
                .repeat(1, num_components, 1)
                .reshape(batch_size * num_components, seq_len)
                .contiguous()
            )
            
        # Apply inter attention
        norm_inter = self.norm1_inter(x_inter.to(self.norm1_inter.weight.dtype))
        out_inter = self.inter_attn(norm_inter, mask=r_mask)
        x_inter = x_inter + out_inter
        
        # Transpose back: [batch_size, num_components, seq_len, embed_dim] -> [batch_size, seq_len, num_components, embed_dim]
        x = x_inter.reshape(batch_size, num_components, seq_len, embed_dim).transpose(1, 2).contiguous()
        
        # 3. Feed Forward Network
        x = x + self.ffn(self.norm2(x.to(self.norm2.weight.dtype)))
        
        return x


class HierarchicalTransformerLayer(nn.Module):
    """
    CLS-augmented flat 1D hierarchical attention.

    Input x_full = [CLS_0, CLS_1, ..., CLS_{T-1}, tok_0, tok_1, ..., tok_N]
    shape: [B, max_T + N, D].

    Three sub-operations per layer:
      1. Intra-time self-attention  — block-diagonal mask, Flash/mem-efficient SDPA
      2. Inter-time attention       — on the CLS slice [B, max_T, D] only
      3. FFN                        — on the full [B, max_T+N, D] sequence
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1_intra = RMSNorm(dim)
        self.intra_attn = FlashAttention(dim, num_heads=heads, attn_drop=dropout, proj_drop=dropout)

        self.norm1_inter = RMSNorm(dim)
        self.inter_attn = FlashAttention(dim, num_heads=heads, attn_drop=dropout, proj_drop=dropout)

        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x_full, full_group_ids, full_seq_mask, inter_mask, max_T):
        """
        x_full:         [B, max_T + N, D]  — CLS tokens first, then sequence tokens
        full_group_ids: [B, max_T + N]     — int64; CLS_t has group_id=t, token_i has its time-step index
        full_seq_mask:  [B, max_T + N]     — float; 1 = valid, 0 = padding
        inter_mask:     [B, max_T]         — float; 1 = time step exists, 0 = absent
        max_T:          int                — number of time steps in this batch
        """
        # === 1. Intra-time Attention (block-diagonal) ===
        # CLS_t and tokens of time-step t form one block; cross-block positions are masked out.
        # mask shape [B, L, L] — handled by FlashAttention's dim==3 branch → [B, 1, L, L]
        same_group = full_group_ids.unsqueeze(2) == full_group_ids.unsqueeze(1)          # [B, L, L]
        both_valid = full_seq_mask.bool().unsqueeze(2) & full_seq_mask.bool().unsqueeze(1)  # [B, L, L]
        intra_mask = same_group & both_valid  # [B, L, L] bool

        norm_full = self.norm1_intra(x_full.to(self.norm1_intra.weight.dtype))
        x_full = x_full + self.intra_attn(norm_full, mask=intra_mask)

        # === 2. Inter-time Attention (CLS tokens only) ===
        cls_tokens = x_full[:, :max_T, :]                                           # [B, max_T, D]
        norm_cls   = self.norm1_inter(cls_tokens.to(self.norm1_inter.weight.dtype))
        cls_delta  = self.inter_attn(norm_cls, mask=inter_mask)                     # inter_mask: [B, max_T]

        # Write updated CLS back and keep the rest of the sequence unchanged
        x_full = torch.cat([cls_tokens + cls_delta, x_full[:, max_T:, :]], dim=1)

        # === 3. FFN on full sequence ===
        x_full = x_full + self.ffn(self.norm2(x_full.to(self.norm2.weight.dtype)))

        return x_full
