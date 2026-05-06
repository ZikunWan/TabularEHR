from typing import Optional

import torch
import torch.nn as nn

from .attention import (
    ComponentTransformerLayer,
    HierarchicalTransformerLayer,
    StandardTransformerLayer,
)
from .config import (
    LongTableEncoder1DConfig,
    LongTableEncoder2DConfig,
    LongTableEncoderHierarchicalConfig,
    LongTableEncoderMemoryConfig,
    _BaseTableEncoderConfig,
)
from .embedding import LongTableEmbedding


class _BaseTableEncoder(nn.Module):
    def __init__(self, config: _BaseTableEncoderConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim

        self.embedding = LongTableEmbedding(
            text_dim=config.text_dim,
            dim=config.dim,
            type_vocab_size=config.type_vocab_size,
            fourier_scales=config.fourier_scales,
        )

    def _format_output(
        self,
        output_features: torch.Tensor,
        seq_mask: Optional[torch.Tensor],
        return_mask: bool,
    ):
        if seq_mask is not None:
            seq_mask = seq_mask.to(dtype=output_features.dtype)
        if return_mask:
            return output_features, seq_mask
        return output_features

    def _truncate_table(
        self,
        item_emb: torch.Tensor,
        unit_emb: torch.Tensor,
        value_emb: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        seq_mask: Optional[torch.Tensor],
        type_ids: Optional[torch.Tensor],
    ):
        max_table_len = self.config.max_table_len
        if max_table_len is None or item_emb.shape[1] <= max_table_len:
            return item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids

        item_emb = item_emb[:, -max_table_len:]
        unit_emb = unit_emb[:, -max_table_len:]
        value_emb = value_emb[:, -max_table_len:]
        times = times[:, -max_table_len:]
        numeric_values = numeric_values[:, -max_table_len:]
        numeric_mask = numeric_mask[:, -max_table_len:]
        if seq_mask is not None:
            seq_mask = seq_mask[:, -max_table_len:]
        if type_ids is not None:
            type_ids = type_ids[:, -max_table_len:]
        return item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids


class LongTableEncoder1D(_BaseTableEncoder):
    def __init__(self, config: LongTableEncoder1DConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            StandardTransformerLayer(
                config.dim,
                config.heads,
                config.dim_head,
                config.mlp_dim,
                config.dropout,
            )
            for _ in range(config.depth)
        ])

    def forward(
        self,
        item_emb: torch.Tensor,
        unit_emb: torch.Tensor,
        value_emb: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ) -> torch.Tensor:
        item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids = self._truncate_table(
            item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids
        )
        if seq_mask is not None:
            seq_mask = seq_mask.to(dtype=item_emb.dtype)

        output_features = self.embedding(
            item_emb,
            unit_emb,
            value_emb,
            times,
            numeric_values,
            numeric_mask,
            type_ids=type_ids,
            use_2d_attention=False,
        )

        for layer in self.layers:
            output_features = layer(output_features, seq_mask, causal=self.config.is_causal)

        return self._format_output(output_features, seq_mask, return_mask)


class LongTableEncoder2D(_BaseTableEncoder):
    def __init__(self, config: LongTableEncoder2DConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            ComponentTransformerLayer(
                config.dim,
                config.heads,
                config.dim_head,
                config.mlp_dim,
                config.dropout,
            )
            for _ in range(config.depth)
        ])

    def forward(
        self,
        item_emb: torch.Tensor,
        unit_emb: torch.Tensor,
        value_emb: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ) -> torch.Tensor:
        item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids = self._truncate_table(
            item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids
        )
        if seq_mask is not None:
            seq_mask = seq_mask.to(dtype=item_emb.dtype)

        item_proj, val_proj, unit_proj, type_emb, time_emb, time_mask = self.embedding(
            item_emb,
            unit_emb,
            value_emb,
            times,
            numeric_values,
            numeric_mask,
            type_ids=type_ids,
            use_2d_attention=True,
        )

        if type_emb is not None:
            component_stack = torch.stack([item_proj, val_proj, unit_proj, type_emb], dim=2)
        else:
            component_stack = torch.stack([item_proj, val_proj, unit_proj], dim=2)

        num_components = component_stack.shape[2]
        time_emb = time_emb.unsqueeze(2).expand(-1, -1, num_components, -1)
        time_mask = time_mask.unsqueeze(2).expand(-1, -1, num_components, -1)
        component_stack = component_stack + time_emb.to(component_stack.dtype) * time_mask.to(component_stack.dtype)

        for layer in self.layers:
            component_stack = layer(component_stack, seq_mask, causal=self.config.is_causal)

        output_features = component_stack.mean(dim=2)
        return self._format_output(output_features, seq_mask, return_mask)


class LongTableEncoderHierarchical(_BaseTableEncoder):
    def __init__(self, config: LongTableEncoderHierarchicalConfig):
        super().__init__(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList([
            HierarchicalTransformerLayer(
                config.dim,
                config.heads,
                config.dim_head,
                config.mlp_dim,
                config.dropout,
            )
            for _ in range(config.depth)
        ])

    def forward(
        self,
        item_emb: torch.Tensor,
        unit_emb: torch.Tensor,
        value_emb: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ) -> torch.Tensor:
        item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids = self._truncate_table(
            item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids
        )
        item_proj, val_proj, unit_proj, type_emb, time_emb, time_mask = self.embedding(
            item_emb,
            unit_emb,
            value_emb,
            times,
            numeric_values,
            numeric_mask,
            type_ids=type_ids,
            use_2d_attention=True,
        )
        batch_size, seq_len, embed_dim = item_proj.shape
        device = item_proj.device

        if seq_mask is None:
            seq_mask = torch.ones(batch_size, seq_len, device=device, dtype=item_emb.dtype)
        else:
            seq_mask = seq_mask.to(dtype=item_emb.dtype)

        x_seq = item_proj + val_proj + unit_proj
        if type_emb is not None:
            x_seq = x_seq + type_emb
        x_seq = x_seq + time_emb.to(x_seq.dtype) * time_mask

        valid_mask = seq_mask.bool()
        is_boundary = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=device),
            times[:, 1:] != times[:, :-1],
        ], dim=1)
        is_boundary = is_boundary & valid_mask
        group_ids = torch.cumsum(is_boundary, dim=1) - 1
        group_ids = group_ids * seq_mask.long()

        num_times_per_batch = (group_ids * seq_mask.long()).max(dim=1).values + 1
        num_times_per_batch = num_times_per_batch * (seq_mask.sum(dim=1) > 0).long()
        max_num_times = num_times_per_batch.max().item()

        if max_num_times == 0:
            output_features = torch.zeros(batch_size, 1, embed_dim, device=device, dtype=item_proj.dtype)
            output_mask = torch.zeros(batch_size, 1, device=device, dtype=item_proj.dtype)
            return self._format_output(output_features, output_mask, return_mask)

        boundary_b_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)[is_boundary]
        boundary_t_idx = group_ids[is_boundary]
        boundary_time_emb = time_emb[is_boundary]

        inter_mask = torch.zeros(batch_size, max_num_times, device=device, dtype=item_proj.dtype)
        inter_mask[boundary_b_idx, boundary_t_idx] = 1.0

        grid_time_emb = torch.zeros(batch_size, max_num_times, embed_dim, device=device, dtype=item_proj.dtype)
        grid_time_emb[boundary_b_idx, boundary_t_idx] = boundary_time_emb.to(item_proj.dtype)
        cls_tokens_full = self.cls_token.expand(batch_size, max_num_times, -1).to(item_proj.dtype) + grid_time_emb

        x_full = torch.cat([cls_tokens_full, x_seq], dim=1)

        cls_group_ids = torch.arange(max_num_times, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        full_group_ids = torch.cat([cls_group_ids, group_ids], dim=1)
        full_seq_mask = torch.cat([inter_mask, seq_mask.to(item_proj.dtype)], dim=1)

        for layer in self.layers:
            x_full = layer(
                x_full,
                full_group_ids,
                full_seq_mask,
                inter_mask,
                max_num_times,
                causal=self.config.is_causal,
            )

        output_features = x_full[:, :max_num_times, :]
        return self._format_output(output_features, inter_mask, return_mask)


class LongTableEncoderMemory(_BaseTableEncoder):
    def __init__(self, config: LongTableEncoderMemoryConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([
            StandardTransformerLayer(
                config.dim,
                config.heads,
                config.dim_head,
                config.mlp_dim,
                config.dropout,
            )
            for _ in range(config.depth)
        ])
        self.memory_norm = nn.LayerNorm(config.dim)
        if config.memory_pooling == "cls":
            self.memory_cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
            nn.init.trunc_normal_(self.memory_cls_token, std=0.02)
        elif config.memory_pooling == "attention":
            self.memory_attention = nn.Linear(config.dim, 1)

    def _pool_memory_chunks(
        self,
        chunk_features: torch.Tensor,
        chunk_mask: torch.Tensor,
        batch_size: int,
        num_chunks: int,
        embed_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.config.memory_pooling == "cls":
            cls_tokens = self.memory_cls_token.expand(chunk_features.shape[0], -1, -1).to(dtype)
            chunk_features = torch.cat([cls_tokens, chunk_features], dim=1)
            cls_mask = torch.ones(chunk_mask.shape[0], 1, device=chunk_mask.device, dtype=chunk_mask.dtype)
            chunk_mask = torch.cat([cls_mask, chunk_mask], dim=1)

        for layer in self.layers:
            chunk_features = layer(chunk_features, chunk_mask)

        if self.config.memory_pooling == "cls":
            memory_tokens = chunk_features[:, 0, :]
            return memory_tokens.view(batch_size, num_chunks, embed_dim)

        if self.config.memory_pooling == "attention":
            scores = self.memory_attention(chunk_features.to(self.memory_attention.weight.dtype)).squeeze(-1)
            scores = scores.masked_fill(chunk_mask == 0, -1e4)
            weights = torch.softmax(scores, dim=-1).to(dtype) * chunk_mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
            memory_tokens = torch.bmm(weights.unsqueeze(1), chunk_features).squeeze(1)
            return memory_tokens.view(batch_size, num_chunks, embed_dim)

        chunk_weight = chunk_mask.unsqueeze(-1)
        memory_tokens = (chunk_features * chunk_weight).sum(dim=1)
        memory_den = chunk_weight.sum(dim=1).clamp(min=1.0)
        memory_tokens = memory_tokens / memory_den
        return memory_tokens.view(batch_size, num_chunks, embed_dim)

    def _build_memory_context(
        self,
        output_features: torch.Tensor,
        seq_mask: Optional[torch.Tensor],
    ):
        batch_size, seq_len, embed_dim = output_features.shape
        device = output_features.device
        dtype = output_features.dtype

        if seq_mask is None:
            seq_mask = torch.ones(batch_size, seq_len, device=device, dtype=dtype)
        else:
            seq_mask = seq_mask.to(dtype=dtype)

        max_recent_len = min(self.config.max_recent_len, seq_len)
        history_len = seq_len - max_recent_len
        recent_features = output_features[:, history_len:, :]
        recent_mask = seq_mask[:, history_len:]

        if history_len <= 0:
            return recent_features, recent_mask

        history_features = output_features[:, :history_len, :]
        history_mask = seq_mask[:, :history_len]

        chunk_len = self.config.memory_chunk_len
        num_chunks = (history_len + chunk_len - 1) // chunk_len
        padded_len = num_chunks * chunk_len
        pad_len = padded_len - history_len
        if pad_len > 0:
            history_features = torch.cat([
                history_features,
                torch.zeros(batch_size, pad_len, embed_dim, device=device, dtype=dtype),
            ], dim=1)
            history_mask = torch.cat([
                history_mask,
                torch.zeros(batch_size, pad_len, device=device, dtype=dtype),
            ], dim=1)

        history_features = history_features.view(batch_size, num_chunks, chunk_len, embed_dim)
        history_mask = history_mask.view(batch_size, num_chunks, chunk_len)
        chunk_features = history_features.reshape(batch_size * num_chunks, chunk_len, embed_dim)
        chunk_mask = history_mask.reshape(batch_size * num_chunks, chunk_len)

        memory_tokens = self._pool_memory_chunks(
            chunk_features,
            chunk_mask,
            batch_size,
            num_chunks,
            embed_dim,
            dtype,
        )
        memory_tokens = self.memory_norm(memory_tokens.to(self.memory_norm.weight.dtype)).to(dtype)
        memory_mask = (history_mask.sum(dim=2) > 0).to(dtype)

        max_memory_tokens = min(self.config.max_memory_tokens, memory_tokens.shape[1])
        memory_tokens = memory_tokens[:, -max_memory_tokens:, :]
        memory_mask = memory_mask[:, -max_memory_tokens:]

        output_features = torch.cat([memory_tokens, recent_features], dim=1)
        seq_mask = torch.cat([memory_mask, recent_mask], dim=1)
        return output_features, seq_mask

    def forward(
        self,
        item_emb: torch.Tensor,
        unit_emb: torch.Tensor,
        value_emb: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        return_mask: bool = False,
    ) -> torch.Tensor:
        item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids = self._truncate_table(
            item_emb, unit_emb, value_emb, times, numeric_values, numeric_mask, seq_mask, type_ids
        )
        if seq_mask is not None:
            seq_mask = seq_mask.to(dtype=item_emb.dtype)

        output_features = self.embedding(
            item_emb,
            unit_emb,
            value_emb,
            times,
            numeric_values,
            numeric_mask,
            type_ids=type_ids,
            use_2d_attention=False,
        )
        output_features, seq_mask = self._build_memory_context(output_features, seq_mask)

        for layer in self.layers:
            output_features = layer(output_features, seq_mask, causal=self.config.is_causal)

        return self._format_output(output_features, seq_mask, return_mask)
