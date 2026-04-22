import torch
import torch.nn as nn
from typing import Optional
from .attention import StandardTransformerLayer, ComponentTransformerLayer
from .adapter import QFormer
from .embedding import LongTableEmbedding
from .config import TableEncoderConfig
import models.TableEncoder.attention as attention


class LongTableEncoder(nn.Module):
    def __init__(self, config: TableEncoderConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.attention_mode = config.attention_mode
        
        # 1. Embedding Layer (Fuses pre-computed Item, Value, Unit embeddings with Time)
        self.embedding = LongTableEmbedding(text_dim=config.text_dim,
                                            dim=config.dim,
                                            type_vocab_size=config.type_vocab_size,
                                            fourier_scales=config.fourier_scales)
        
        # 2. Transformer Backbone
        if self.attention_mode == 'hierarchical':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.hierarchical_layers = nn.ModuleList([
                attention.HierarchicalTransformerLayer(config.dim, config.heads, config.dim_head, config.mlp_dim, config.dropout)
                for _ in range(config.depth)
            ])
        elif self.attention_mode == '2d_grid':
            self.layers = nn.ModuleList([
                ComponentTransformerLayer(config.dim, config.heads, config.dim_head, config.mlp_dim, config.dropout)
                for _ in range(config.depth)
            ])
        else:
            self.layers = nn.ModuleList([
                StandardTransformerLayer(config.dim, config.heads, config.dim_head, config.mlp_dim, config.dropout)
                for _ in range(config.depth)
            ])
        
        # 3. Q-Former Adapter
        self.qformer = QFormer(config.dim, config.num_queries, dim_out=config.dim_out)

    def forward(self, 
                item_emb: torch.Tensor,
                unit_emb: torch.Tensor,
                value_emb: torch.Tensor,
                times: torch.Tensor,
                numeric_values: torch.Tensor,
                numeric_mask: torch.Tensor,
                seq_mask: Optional[torch.Tensor] = None,
                type_ids: Optional[torch.Tensor] = None):
        """
        Forward pass with pre-computed embeddings.
        
        Args:
            item_emb: [batch_size, seq_len, text_dim] - Pre-computed item embeddings
            unit_emb: [batch_size, seq_len, text_dim] - Pre-computed unit embeddings
            value_emb: [batch_size, seq_len, text_dim] - Pre-computed value text embeddings
            times: [batch_size, seq_len] - Time values
            numeric_values: [batch_size, seq_len] - Numeric values
            numeric_mask: [batch_size, seq_len] - 1 for numeric, 0 for text
            seq_mask: [batch_size, seq_len] - 1 for valid, 0 for padding
            type_ids: [batch_size, seq_len] - Type Category IDs (Optional)
        
        Returns:
            query_embeddings: [batch_size, num_queries, embed_dim]
        """
        if seq_mask is not None:
            seq_mask = seq_mask.to(dtype=item_emb.dtype)

        if self.attention_mode == '1d':
            # 1. Get Sequence Embeddings (1D shape: [batch_size, seq_len, embed_dim])
            output_features = self.embedding(item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, type_ids=type_ids, use_2d_attention=False)
            
            # Standard 1D Transformer Backbone
            for layer in self.layers:
                output_features = layer(output_features, seq_mask)
        elif self.attention_mode == '2d_grid':
            # 1. Get Separated Embeddings
            item_proj, val_proj, unit_proj, type_emb, time_emb, time_mask = self.embedding(
                item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, type_ids=type_ids, use_2d_attention=True
            )
            batch_size, seq_len, embed_dim = item_proj.shape
            
            # Stack the components: Item, Value, Unit, Type
            # [batch_size, seq_len, num_components, embed_dim]
            if type_emb is not None:
                component_stack = torch.stack([item_proj, val_proj, unit_proj, type_emb], dim=2)
            else:
                component_stack = torch.stack([item_proj, val_proj, unit_proj], dim=2)
                
            num_components = component_stack.shape[2]
            
            # Add time embeddings to all components so they carry temporal context
            time_emb_expanded = time_emb.unsqueeze(2).expand(-1, -1, num_components, -1).contiguous().to(dtype=component_stack.dtype)
            time_mask_expanded = time_mask.unsqueeze(2).expand(-1, -1, num_components, -1).contiguous().to(dtype=component_stack.dtype)
            
            component_stack = component_stack + time_emb_expanded * time_mask_expanded
            
            for layer in self.layers:
                component_stack = layer(component_stack, seq_mask)
                    
            # Pool the components back to a 1D sequence for the Q-Former
            output_features = component_stack.mean(dim=2) # [batch_size, seq_len, embed_dim]
            
        elif self.attention_mode == 'hierarchical':
            # 1. Get Separated Embeddings
            item_proj, val_proj, unit_proj, type_emb, time_emb, time_mask = self.embedding(
                item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, type_ids=type_ids, use_2d_attention=True
            )
            batch_size, seq_len, embed_dim = item_proj.shape
            device = item_proj.device

            # Form flat token sequence (include time positional encoding in each token)
            x_seq = item_proj + val_proj + unit_proj
            if type_emb is not None:
                x_seq = x_seq + type_emb
            x_seq = x_seq + time_emb.to(x_seq.dtype) * time_mask

            valid_mask = seq_mask.bool()

            # 2. GPU-native time grouping (vectorized, no Python loops)
            is_boundary = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                times[:, 1:] != times[:, :-1]
            ], dim=1)
            is_boundary = is_boundary & valid_mask
            group_ids = torch.cumsum(is_boundary, dim=1) - 1  # [B, N], 0-indexed
            group_ids = group_ids * seq_mask.long()

            num_times_per_batch = (group_ids * seq_mask.long()).max(dim=1).values + 1
            num_times_per_batch = num_times_per_batch * (seq_mask.sum(dim=1) > 0).long()
            max_num_times = num_times_per_batch.max().item()

            if max_num_times == 0:
                out = self.qformer(torch.zeros(batch_size, 1, embed_dim, device=device, dtype=item_proj.dtype),
                                   torch.zeros(batch_size, 1, device=device, dtype=item_proj.dtype))
                return out

            # 3. Compute inter_mask and per-time-step time embeddings for CLS init
            boundary_b_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)[is_boundary]
            boundary_t_idx = group_ids[is_boundary]
            boundary_time_emb = time_emb[is_boundary]

            inter_mask = torch.zeros(batch_size, max_num_times, device=device, dtype=item_proj.dtype)
            inter_mask[boundary_b_idx, boundary_t_idx] = 1.0

            # 4. Build CLS tokens: learnable cls_token + time positional encoding per time step
            grid_time_emb = torch.zeros(batch_size, max_num_times, embed_dim, device=device, dtype=item_proj.dtype)
            grid_time_emb[boundary_b_idx, boundary_t_idx] = boundary_time_emb.to(item_proj.dtype)
            cls_tokens_full = self.cls_token.expand(batch_size, max_num_times, -1).to(item_proj.dtype) + grid_time_emb
            # cls_tokens_full: [B, max_T, D]

            # 5. Build full flat sequence: [CLS_0 ... CLS_{T-1}, tok_0 ... tok_N]
            x_full = torch.cat([cls_tokens_full, x_seq], dim=1)  # [B, max_T + N, D]

            # 6. Build full group_ids: CLS_t → group_id=t, tokens keep their group_id
            cls_group_ids = torch.arange(max_num_times, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
            full_group_ids = torch.cat([cls_group_ids, group_ids], dim=1)  # [B, max_T + N]

            # 7. Build full seq_mask
            full_seq_mask = torch.cat([inter_mask, seq_mask.to(item_proj.dtype)], dim=1)  # [B, max_T + N]

            # 8. Apply transformer layers (flat 1D, no 4D grid)
            for layer in self.hierarchical_layers:
                x_full = layer(x_full, full_group_ids, full_seq_mask, inter_mask, max_num_times)

            # 9. The CLS tokens (first max_T positions) are the final time-step representations
            output_features = x_full[:, :max_num_times, :]  # [B, max_T, D]
            seq_mask = inter_mask


        # 3. Q-Former Adapter
        out = self.qformer(output_features, seq_mask.to(dtype=item_emb.dtype) if seq_mask is not None else None)
        
        return out
