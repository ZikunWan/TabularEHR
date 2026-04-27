import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class TextEncoder(nn.Module):

    def __init__(
        self, model_name_or_path: str, embed_dim: int = 768, freeze_bert: bool = False
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path, output_hidden_states=True)
        self.bert_model = AutoModel.from_pretrained(model_name_or_path, config=config)
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        hidden_size = getattr(config, "hidden_size", embed_dim)
        self.hidden_size = hidden_size
        self.mlp_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self._init_projection(hidden_size)
        self.ensure_parameters_contiguous()

    def _init_projection(self, hidden_size: int) -> None:
        nn.init.constant_(self.logit_scale, math.log(1 / 0.07))
        for layer in self.mlp_embed:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=hidden_size**-0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def encode_text(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        output = self.bert_model(**model_inputs)
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        return self.mlp_embed(pooled)

    def forward(
        self, text1: Dict[str, torch.Tensor], text2: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text1_features = F.normalize(self.encode_text(text1), dim=-1)
        text2_features = F.normalize(self.encode_text(text2), dim=-1)
        return text1_features, text2_features, self.logit_scale.exp()

    def ensure_parameters_contiguous(self) -> None:
        with torch.no_grad():
            for name, param in self.named_parameters():
                if not param.data.is_contiguous():
                    param.data = param.data.contiguous()

class KnowledgeEncoderForTrainer(TextEncoder):
    def forward(
        self,
        name_input_ids: torch.Tensor,
        name_attention_mask: torch.Tensor,
        definition_input_ids: torch.Tensor,
        definition_attention_mask: torch.Tensor,
        name_token_type_ids: Optional[torch.Tensor] = None,
        definition_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        name = {
            "input_ids": name_input_ids,
            "attention_mask": name_attention_mask,
            "token_type_ids": name_token_type_ids,
        }
        definition = {
            "input_ids": definition_input_ids,
            "attention_mask": definition_attention_mask,
            "token_type_ids": definition_token_type_ids,
        }
        name_features, definition_features, scale = super().forward(name, definition)
        logits = scale * name_features @ definition_features.t()
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        )
        return {"loss": loss}


class KnowledgeGraphEncoderForTrainer(TextEncoder):
    def __init__(
        self,
        model_name_or_path: str,
        num_relations: int,
        margin: float = 1.0,
        distance_p: int = 2,
        relation_reg: float = 1e-4,
        freeze_bert: bool = False,
    ):
        super().__init__(model_name_or_path, freeze_bert=freeze_bert)
        self.margin = margin
        self.distance_p = distance_p
        self.relation_reg = relation_reg
        self.relation_embeddings = nn.Embedding(num_relations, self.hidden_size)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        self.ensure_parameters_contiguous()
    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return F.normalize(self.encode_text(batch), dim=-1)

    def _distance(
        self, head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.vector_norm(head + relation - tail, ord=self.distance_p, dim=-1)

    def forward(
        self,
        head_input_ids: torch.Tensor,
        head_attention_mask: torch.Tensor,
        tail_input_ids: torch.Tensor,
        tail_attention_mask: torch.Tensor,
        relation_ids: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_attention_mask: torch.Tensor,
        negative_is_head: torch.Tensor,
        head_token_type_ids: Optional[torch.Tensor] = None,
        tail_token_type_ids: Optional[torch.Tensor] = None,
        negative_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        head = self._encode(head_input_ids, head_attention_mask, head_token_type_ids)
        tail = self._encode(tail_input_ids, tail_attention_mask, tail_token_type_ids)
        relation = self.relation_embeddings(relation_ids)

        batch_size, num_negatives, seq_len = negative_input_ids.shape
        flat_negative_token_type_ids = None
        if negative_token_type_ids is not None:
            flat_negative_token_type_ids = negative_token_type_ids.view(
                batch_size * num_negatives, seq_len
            )
        negative = self._encode(
            negative_input_ids.view(batch_size * num_negatives, seq_len),
            negative_attention_mask.view(batch_size * num_negatives, seq_len),
            flat_negative_token_type_ids,
        ).view(batch_size, num_negatives, -1)

        pos_dist = self._distance(head, relation, tail)
        corrupt_head = negative_is_head.bool().unsqueeze(-1)
        neg_head = torch.where(corrupt_head, negative, head.unsqueeze(1))
        neg_tail = torch.where(corrupt_head, tail.unsqueeze(1), negative)
        neg_dist = self._distance(neg_head, relation.unsqueeze(1), neg_tail)

        loss = F.relu(self.margin + pos_dist.unsqueeze(1) - neg_dist)
        loss = loss.mean()
        if self.relation_reg > 0:
            loss = loss + self.relation_reg * relation.pow(2).mean()

        violation_rate = (pos_dist.unsqueeze(1) + self.margin > neg_dist).float().mean()
        return {
            "loss": loss,
            "pos_dist": pos_dist.mean(),
            "neg_dist": neg_dist.mean(),
            "margin_violation_rate": violation_rate,
        }
