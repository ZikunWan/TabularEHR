from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D
from models.query_attention import QueryCrossAttentionHead


@dataclass
class CandidateDecoderOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    scores: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class TaskQueryCandidateDecoderModel(PreTrainedModel):
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
        self.answer_projection = nn.Linear(query_dim, query_dim)
        self.candidate_projection = nn.Linear(query_dim, query_dim)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
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
        candidate_embeds: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
        type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CandidateDecoderOutput:
        query_state = self.extract_features(
            item_ids=item_ids,
            unit_ids=unit_ids,
            value_text_ids=value_text_ids,
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            query_embeds=query_embeds,
            seq_mask=seq_mask,
            type_ids=type_ids,
        )
        answer_state = F.normalize(self.answer_projection(query_state), dim=-1)
        candidate_state = F.normalize(self.candidate_projection(candidate_embeds.to(query_state.dtype)), dim=-1)
        if answer_state.dim() == 2:
            scores = torch.einsum("bd,bkd->bk", answer_state, candidate_state)
        else:
            scores = torch.einsum("bqd,bqkd->bqk", answer_state, candidate_state)
        scores = scores * self.logit_scale.exp().clamp(max=100.0)
        if candidate_mask is not None:
            scores = scores.masked_fill(candidate_mask <= 0, torch.finfo(scores.dtype).min)

        loss = None
        if labels is not None:
            if scores.dim() == 2:
                loss = F.cross_entropy(scores.float(), labels.long())
            else:
                valid_mask = labels != -100
                if valid_mask.any():
                    loss = F.cross_entropy(scores.float()[valid_mask], labels.long()[valid_mask])
                else:
                    loss = scores.sum() * 0.0

        return CandidateDecoderOutput(loss=loss, scores=scores, logits=scores)

    def extract_features(
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
    ) -> torch.Tensor:
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
        return self.query_head(query_embeds, hidden_states, None)
