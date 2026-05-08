from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .config import LongTableEncoder1DConfig
from .encoder import LongTableEncoder1D


class QueryCrossAttentionHead(nn.Module):
    def __init__(self, config: LongTableEncoder1DConfig, query_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, config.dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.dim,
            num_heads=config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.dim),
        )
        self.norm2 = nn.LayerNorm(config.dim)

    def forward(
        self,
        query_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query = self.query_proj(query_embeds).unsqueeze(1)
        key_padding_mask = None
        if seq_mask is not None:
            key_padding_mask = ~seq_mask.bool()

        attn_out, _ = self.cross_attn(
            query=query,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
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
        self.text_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.query_head = QueryCrossAttentionHead(config, query_dim=query_dim)
        self.num_classes = config.num_classes
        self.problem_type = config.problem_type
        self.classifier = nn.Linear(config.dim, self.num_classes)
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
        pooled = self.query_head(query_embeds, hidden_states, hidden_mask)
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
                loss_fct = nn.BCEWithLogitsLoss(reduction="none")
                mask = labels != -100
                safe_labels = labels.clone()
                safe_labels[~mask] = 0
                loss_matrix = loss_fct(logits, safe_labels.to(logits.dtype))
                loss = (loss_matrix * mask.to(logits.dtype)).sum() / mask.sum().clamp(min=1)
            else:
                raise ValueError(f"Unsupported problem_type: {self.problem_type}")

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
