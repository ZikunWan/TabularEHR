from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .adapter import QFormerAdapter
from .attention import CrossFlashAttention, RMSNorm
from .config import LongTableEncoder1DConfig
from .encoder import LongTableEncoder1D


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
        query = self.query_norm(query_embeds.to(dtype=hidden_states.dtype)).unsqueeze(1)
        hidden_states = self.context_norm(hidden_states)
        attn_out = self.cross_attn(query, hidden_states, seq_mask)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))
        return query.squeeze(1)


class TaskQueryClassificationModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config: LongTableEncoder1DConfig,
        embedding_matrix: torch.Tensor,
        query_dim: int,
    ):
        super().__init__(config)
        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.text_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.query_head = QueryCrossAttentionHead(config, query_dim=query_dim)
        self.num_classes = config.num_classes
        self.problem_type = config.problem_type
        self.classifier = nn.Linear(query_dim, self.num_classes)
        self.post_init()

    def _init_weights(self, module):
        if module is self.text_embedding:
            return
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        item_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        value_text_ids: torch.Tensor,
        times: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        query_embeds: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        item_emb = self.text_embedding(item_ids)
        unit_emb = self.text_embedding(unit_ids)
        value_emb = self.text_embedding(value_text_ids)

        hidden_states, hidden_mask = self.encoder(
            item_emb=item_emb,
            unit_emb=unit_emb,
            value_emb=value_emb,
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            seq_mask=seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )
        hidden_states = self.adapter(hidden_states, hidden_mask)
        pooled = self.query_head(query_embeds, hidden_states, None)
        logits = self.classifier(pooled)
        if self.problem_type == "multi_label_classification" and hasattr(self.config, "num_points") and hasattr(self.config, "num_metrics"):
            logits = logits.view(-1, self.config.num_points, self.config.num_metrics)

        loss = None
        if labels is not None:
            if self.problem_type == "single_label_classification" and self.num_classes == 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).to(logits.dtype))
            elif self.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1).long())
            elif self.problem_type == "multi_label_classification":
                mask = labels != -100
                if mask.any():
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits[mask], labels[mask].to(logits.dtype))
                else:
                    loss = logits[mask].sum() * 0.0
            else:
                raise ValueError(f"Unsupported problem_type: {self.problem_type}")

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
