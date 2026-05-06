import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .TableEncoder.adapter import QFormerAdapter
from .TableEncoder.config import LongTableEncoder1DConfig
from .TableEncoder.encoder import LongTableEncoder1D


class LongTableEncoderClassifier(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"
    
    def __init__(self, config: LongTableEncoder1DConfig):
        super().__init__(config)
        
        self.encoder = LongTableEncoder1D(config=config)
        self.adapter = QFormerAdapter(config)
        self.num_classes = config.num_classes
        self.problem_type = config.problem_type
        classifier_dim = config.dim_out if config.dim_out is not None else config.dim
        self.classifier = nn.Linear(classifier_dim, self.num_classes)
            
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
    def forward(self,
                item_emb: torch.Tensor,
                unit_emb: torch.Tensor,
                value_emb: torch.Tensor,
                times: torch.Tensor,
                numeric_values: torch.Tensor,
                numeric_mask: torch.Tensor,
                seq_mask: torch.Tensor = None,
                type_ids: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs): # Catch extra kwargs from Trainer
        """
        Forward pass with pre-computed embeddings.
        """
        hidden_states, hidden_mask = self.encoder(
            item_emb, unit_emb, value_emb,
            times, numeric_values, numeric_mask,
            seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )
        # [Batch, NumQueries, Dim]
        x = self.adapter(hidden_states, hidden_mask)
        x_pooled = x.mean(dim=1)
        logits = self.classifier(x_pooled)
        
        # Reshape dynamically for multi-label tasks structured across points and metrics
        if self.problem_type == "multi_label_classification":
            # Target User Layout: [batch_size, 时间窗数 (points), 指标数 (metrics)]
            logits = logits.view(-1, self.config.num_points, self.config.num_metrics)

        loss = None
        if labels is not None:
            if self.problem_type == "single_label_classification":
                if self.num_classes == 1:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1).to(logits.dtype))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                mask = (labels != -100)
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
