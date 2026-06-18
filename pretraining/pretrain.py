import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, Dataset, Subset
from tqdm.auto import tqdm
from transformers import (
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.TableEncoder.adapter import QFormerAdapter
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.encoder import LongTableEncoder1D
from models.TableEncoder.next_token_decoder import NextTokenPredictionDecoder
from models.TableEncoder.query_classifier import QueryCrossAttentionHead
from utils.metrics import compute_classification_metrics

try:
    from . import next_token_prediction as ntp
    from . import phenotype_metric_learning as pml
    from . import task_query_classification as tqc
except ImportError:
    import next_token_prediction as ntp
    import phenotype_metric_learning as pml
    import task_query_classification as tqc


@dataclass
class DataArguments:
    dataset: List[str] = field(
        default_factory=lambda: ["mimic_iv", "eicu", "ehrshot"]
    )
    root_dir: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular"
    )
    eicu_root_dir: str = field(default="/data/zikun_workspace/eicu-crd")
    eicu_processed_dir: str = field(
        default="/data/zikun_workspace/eicu-crd/processed"
    )
    ehrshot_root_dir: str = field(default="/data/EHR_data_public/EHRSHOT")
    table_text_embedding: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/.cache/embeddings/mimic_iv/"
            "text_embeddings_stage2.pt"
        ]
    )
    eicu_table_text_embedding: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/.cache/embeddings/eicu/"
            "text_embeddings_stage2.pt"
        ]
    )
    ehrshot_table_text_embedding: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/.cache/embeddings/ehrshot/"
            "text_embeddings_stage2.pt"
        ]
    )
    type_vocab_file: str = field(
        default="/data/zikun_workspace/code/data/type_vocab.json"
    )

    ntp_train_info_path: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/"
            "next_token_prediction.csv"
        ]
    )
    ntp_val_info_path: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val/"
            "next_token_prediction.csv"
        ]
    )
    eicu_ntp_train_info_path: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/eicu-crd/processed/pretraining_index/"
            "sample_info_train.json"
        ]
    )
    eicu_ntp_val_info_path: List[str] = field(
        default_factory=lambda: [
            "/data/zikun_workspace/eicu-crd/processed/pretraining_index/"
            "sample_info_val.json"
        ]
    )
    ehrshot_ntp_train_info_path: List[str] = field(
        default_factory=lambda: [
            "/data/EHR_data_public/EHRSHOT/pretraining_index/"
            "sample_info_train.csv"
        ]
    )
    ehrshot_ntp_val_info_path: List[str] = field(
        default_factory=lambda: [
            "/data/EHR_data_public/EHRSHOT/pretraining_index/"
            "sample_info_val.csv"
        ]
    )

    task_train_sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train"
    )
    task_val_sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val"
    )
    eicu_task_train_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/"
        "sample_info_train.json"
    )
    eicu_task_val_sample_info_path: str = field(
        default="/data/zikun_workspace/eicu-crd/processed/"
        "sample_info_val.json"
    )
    ehrshot_task_train_sample_info_path: str = field(
        default="/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv"
    )
    ehrshot_task_val_sample_info_path: str = field(
        default="/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv"
    )
    task_query_embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/pretrain/"
        "task_query_knowledge_embeddings.pt"
    )

    phenotype_spec_path: str = field(
        default="/data/zikun_workspace/.cache/phenotype_metric_learning/"
        "phenotype_query_specs.json"
    )
    phenotype_query_embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/pretrain/"
        "phenotype_query_knowledge_embeddings.pt"
    )
    phenotype_preprocessed_input_dir: str = field(
        default="/data/zikun_workspace/.cache/phenotype_metric_learning/inputs"
    )

    knowledge_encoder_path: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/"
        "knowledge_encoder/clinicalBERT_after_stage2/best.pt"
    )
    knowledge_encoder_base_model_path: str = field(
        default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
    )
    query_max_length: int = field(default=128)
    query_embedding_batch_size: int = field(default=256)
    max_table_len: Optional[int] = field(default=4096)
    min_table_rows: int = field(default=2)
    max_ntp_train_samples: Optional[int] = field(default=None)
    max_ntp_eval_samples: Optional[int] = field(default=None)
    max_task_train_samples: Optional[int] = field(default=320000)
    max_task_eval_samples: Optional[int] = field(default=None)
    max_metric_train_samples: Optional[int] = field(default=None)
    max_metric_eval_samples: Optional[int] = field(default=None)


@dataclass
class PretrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/joint_pretrain"
    )
    num_train_epochs: float = field(default=5)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=100)
    weight_decay: float = field(default=0.01)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    eval_steps: int = field(default=200)
    save_total_limit: int = field(default=1)
    bf16: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="wandb")
    wandb_project: Optional[str] = field(default="Joint_Pretraining")
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    early_stopping_patience: int = field(default=10)
    ntp_loss_weight: float = field(default=1.0)
    task_loss_weight: float = field(default=1.0)
    metric_loss_weight: float = field(default=1.0)
    huber_delta: float = field(default=1.0)
    projection_loss_weight: float = field(default=1.0)
    transe_loss_weight: float = field(default=0.0)
    relation_l2_weight: float = field(default=0.0)
    min_pair_delta: float = field(default=0.0)
    min_lr_ratio: float = field(default=0.1)

    def __post_init__(self):
        super().__post_init__()
        weights = (
            self.ntp_loss_weight,
            self.task_loss_weight,
            self.metric_loss_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Joint pretraining loss weights must be non-negative.")
        if self.huber_delta <= 0:
            raise ValueError("Huber delta must be positive.")
        metric_weights = (
            self.projection_loss_weight,
            self.transe_loss_weight,
            self.relation_l2_weight,
        )
        if any(weight < 0 for weight in metric_weights):
            raise ValueError("Metric learning loss weights must be non-negative.")
        if self.projection_loss_weight <= 0 and self.transe_loss_weight <= 0:
            raise ValueError(
                "At least one metric learning loss weight must be positive."
            )
        if self.min_pair_delta < 0:
            raise ValueError("Minimum pair delta must be non-negative.")
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        self.eval_strategy = "steps"
        self.load_best_model_at_end = True


class CycledMultiTaskDataset(Dataset):
    def __init__(self, ntp_dataset, task_dataset, metric_dataset):
        self.datasets = {
            "ntp": ntp_dataset,
            "task": task_dataset,
            "metric": metric_dataset,
        }
        lengths = {name: len(dataset) for name, dataset in self.datasets.items()}
        if any(length <= 0 for length in lengths.values()):
            raise ValueError(f"All joint datasets must be non-empty: {lengths}")
        self.lengths = lengths
        self._length = max(lengths.values())

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return {
            name: dataset[index % self.lengths[name]]
            for name, dataset in self.datasets.items()
        }


class MultiTaskCollator:
    def __init__(self, ntp_collator, task_collator, metric_collator):
        self.collators = {
            "ntp": ntp_collator,
            "task": task_collator,
            "metric": metric_collator,
        }

    def __call__(self, batch):
        return {
            name: collator([sample[name] for sample in batch])
            for name, collator in self.collators.items()
        }


class WeightedLossCombiner(nn.Module):
    TASK_NAMES = ("ntp", "task", "metric")

    def __init__(self, weights: List[float]):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float))

    def forward(self, losses: Dict[str, torch.Tensor]):
        loss_vector = torch.stack([losses[name] for name in self.TASK_NAMES])
        weights = self.weights.to(loss_vector.device, loss_vector.dtype)
        weighted_losses = weights * loss_vector
        total = weighted_losses.sum()
        return total, weighted_losses


class JointPretrainingModel(PreTrainedModel):
    config_class = LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config,
        embedding_matrix: torch.Tensor,
        phenotype_query_embedding_matrix: torch.Tensor,
        phenotype_scales: torch.Tensor,
        training_args: PretrainingArguments,
    ):
        super().__init__(config)
        if embedding_matrix.size(1) != config.text_dim:
            raise ValueError("Table embedding dimension does not match config.")
        hidden_size = config.dim_out if config.dim_out is not None else config.dim
        query_dim = hidden_size

        self.encoder = LongTableEncoder1D(config)
        self.adapter = QFormerAdapter(config)
        self.text_embedding_matrix = embedding_matrix.cpu()
        self.ntp_head = NextTokenPredictionDecoder(
            hidden_dim=config.dim,
            text_dim=config.text_dim,
            type_vocab_size=config.type_vocab_size,
            fourier_scales=config.fourier_scales,
        )
        self.task_query_head = QueryCrossAttentionHead(config, query_dim=query_dim)
        self.task_classifier = nn.Linear(query_dim, 1)
        self.metric_pooling = pml.AttentionPooling(hidden_size)
        self.query_embedding_matrix = nn.Parameter(
            phenotype_query_embedding_matrix.float(), requires_grad=False
        )
        self.phenotype_scales = nn.Parameter(
            phenotype_scales.float(), requires_grad=False
        )
        self.relation_projection = nn.Sequential(
            nn.Linear(phenotype_query_embedding_matrix.size(-1), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.huber_delta = float(training_args.huber_delta)
        self.projection_loss_weight = float(
            training_args.projection_loss_weight
        )
        self.transe_loss_weight = float(training_args.transe_loss_weight)
        self.relation_l2_weight = float(training_args.relation_l2_weight)
        self.min_pair_delta = float(training_args.min_pair_delta)
        self.loss_combiner = WeightedLossCombiner(
            weights=[
                training_args.ntp_loss_weight,
                training_args.task_loss_weight,
                training_args.metric_loss_weight,
            ]
        )
        self.post_init()

    def text_lookup(self, token_ids, dtype, device):
        flat = self.text_embedding_matrix.index_select(
            0, token_ids.reshape(-1).cpu()
        )
        flat = flat.to(device=device, dtype=dtype, non_blocking=True)
        return flat.view(*token_ids.shape, flat.size(-1))

    def encode_rows(self, inputs):
        dtype = self.encoder.embedding.item_proj.weight.dtype
        device = self.encoder.embedding.item_proj.weight.device
        item_emb = self.text_lookup(inputs["item_ids"], dtype, device)
        unit_emb = self.text_lookup(inputs["unit_ids"], dtype, device)
        value_emb = self.text_lookup(inputs["value_text_ids"], dtype, device)
        hidden_states, hidden_mask = self.encoder(
            item_emb=item_emb,
            unit_emb=unit_emb,
            value_emb=value_emb,
            times=inputs["times"],
            numeric_values=inputs["numeric_values"],
            numeric_mask=inputs["numeric_mask"],
            seq_mask=inputs.get("seq_mask"),
            type_ids=inputs.get("type_ids"),
            return_mask=True,
        )
        return hidden_states, hidden_mask, item_emb, unit_emb, value_emb

    def forward_ntp(self, inputs):
        hidden_states, hidden_mask, item_emb, unit_emb, value_emb = (
            self.encode_rows(inputs)
        )
        return self.ntp_head(
            hidden_states=hidden_states,
            attention_mask=hidden_mask,
            target_item_emb=item_emb,
            target_unit_emb=unit_emb,
            target_value_text_emb=value_emb,
            target_numeric_values=inputs["numeric_values"],
            target_numeric_mask=inputs["numeric_mask"],
            target_type_ids=inputs["type_ids"],
        )

    def forward_task(self, inputs):
        hidden_states, hidden_mask, _, _, _ = self.encode_rows(inputs)
        adapted = self.adapter(hidden_states, hidden_mask)
        pooled = self.task_query_head(
            inputs["query_embeds"], adapted, None
        )
        logits = self.task_classifier(pooled).view(-1)
        labels = inputs["labels"].view(-1).to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return {"loss": loss, "logits": logits, "labels": labels}

    def relation_vectors(self, dtype: torch.dtype, device: torch.device):
        query_embeddings = self.query_embedding_matrix.to(
            device=device, dtype=dtype
        )
        return F.normalize(self.relation_projection(query_embeddings), dim=-1)

    def _delta_scales(self, global_values, global_mask):
        configured_scales = self.phenotype_scales.to(
            global_values.device, global_values.dtype
        )
        observed_count = global_mask.float().sum(dim=0)
        safe_count = observed_count.clamp_min(1.0)
        mean = (
            global_values.masked_fill(~global_mask, 0.0).sum(dim=0)
            / safe_count
        )
        centered = (global_values - mean).masked_fill(~global_mask, 0.0)
        batch_scale = torch.sqrt(
            centered.pow(2).sum(dim=0) / safe_count
        ).clamp_min(1e-6)
        return torch.where(configured_scales > 0, configured_scales, batch_scale)

    @staticmethod
    def _huber(error: torch.Tensor, delta: float):
        abs_error = error.abs()
        return torch.where(
            abs_error <= delta,
            0.5 * error.pow(2),
            delta * (abs_error - 0.5 * delta),
        )

    def forward_metric(self, inputs):
        phenotype_values = inputs["phenotype_values"]
        phenotype_mask = inputs["phenotype_mask"]
        table_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"phenotype_values", "phenotype_mask", "labels"}
        }
        hidden_states, hidden_mask, _, _, _ = self.encode_rows(table_inputs)
        adapted = self.adapter(hidden_states, hidden_mask)
        pooled_mask = torch.ones(
            adapted.shape[:2],
            dtype=hidden_mask.dtype,
            device=hidden_mask.device,
        )
        local_embeddings = F.normalize(
            self.metric_pooling(adapted, pooled_mask), dim=-1
        )
        global_embeddings = pml.all_gather_with_grad(local_embeddings)
        global_values = pml.all_gather_tensor(
            phenotype_values.to(
                local_embeddings.device, local_embeddings.dtype
            )
        )
        global_mask = pml.all_gather_tensor(
            phenotype_mask.to(local_embeddings.device).bool()
        )
        local_values = phenotype_values.to(
            local_embeddings.device, local_embeddings.dtype
        )
        local_mask = phenotype_mask.to(local_embeddings.device).bool()

        relations = self.relation_vectors(
            local_embeddings.dtype, local_embeddings.device
        )
        delta_embeddings = global_embeddings.unsqueeze(0) - local_embeddings.unsqueeze(1)
        pred_delta = torch.einsum("bgd,qd->bgq", delta_embeddings, relations)

        scales = self._delta_scales(global_values, global_mask)
        true_delta = (
            global_values.unsqueeze(0) - local_values.unsqueeze(1)
        ) / scales.view(1, 1, -1)
        pair_mask = local_mask.unsqueeze(1) & global_mask.unsqueeze(0)
        if self.min_pair_delta > 0:
            pair_mask = pair_mask & (true_delta.abs() >= self.min_pair_delta)

        local_batch_size = local_embeddings.size(0)
        start = pml.gather_batch_start(local_batch_size, local_embeddings.device)
        row_indices = torch.arange(local_batch_size, device=local_embeddings.device)
        self_mask = torch.zeros(
            local_batch_size,
            global_embeddings.size(0),
            dtype=torch.bool,
            device=local_embeddings.device,
        )
        self_mask[row_indices, start + row_indices] = True
        pair_mask = pair_mask & (~self_mask.unsqueeze(-1))

        pair_count = pair_mask.float().sum()
        if pair_count <= 0:
            zero = local_embeddings.sum() * 0.0 + relations.sum() * 0.0
            return {
                "loss": zero,
                "loss_sum": zero.detach(),
                "abs_error_sum": zero.detach(),
                "squared_error_sum": zero.detach(),
                "pair_count": zero.detach(),
            }

        projection_error = pred_delta - true_delta
        projection_terms = self._huber(projection_error, self.huber_delta)
        projection_loss_sum = projection_terms[pair_mask].sum()
        projection_loss = projection_loss_sum / pair_count

        loss = self.projection_loss_weight * projection_loss
        loss_sum = self.projection_loss_weight * projection_loss_sum

        if self.transe_loss_weight > 0:
            transe_target = true_delta.unsqueeze(-1) * relations.view(
                1, 1, relations.size(0), relations.size(1)
            )
            transe_error = delta_embeddings.unsqueeze(2) - transe_target
            transe_terms = transe_error.pow(2).mean(dim=-1)
            transe_loss_sum = transe_terms[pair_mask].sum()
            transe_loss = transe_loss_sum / pair_count
            loss = loss + self.transe_loss_weight * transe_loss
            loss_sum = loss_sum + self.transe_loss_weight * transe_loss_sum

        if self.relation_l2_weight > 0:
            loss = loss + self.relation_l2_weight * relations.pow(2).mean()

        abs_error_sum = projection_error[pair_mask].abs().sum()
        squared_error_sum = projection_error[pair_mask].pow(2).sum()
        return {
            "loss": loss,
            "loss_sum": loss_sum.detach(),
            "abs_error_sum": abs_error_sum.detach(),
            "squared_error_sum": squared_error_sum.detach(),
            "pair_count": pair_count.detach(),
        }

    def forward(
        self,
        objective: str,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if objective == "ntp":
            return self.forward_ntp(inputs)
        if objective == "task":
            return self.forward_task(inputs)
        if objective == "metric":
            return self.forward_metric(inputs)
        if objective == "joint":
            ntp_output = self.forward_ntp(inputs["ntp"])
            task_output = self.forward_task(inputs["task"])
            metric_output = self.forward_metric(inputs["metric"])
            raw_losses = {
                "ntp": ntp_output.loss,
                "task": task_output["loss"],
                "metric": metric_output["loss"],
            }
            total, weighted_losses = self.loss_combiner(raw_losses)
            return {
                "loss": total,
                "weighted_losses": weighted_losses,
                "ntp_output": ntp_output,
                "task_output": task_output,
                "metric_output": metric_output,
            }
        if objective == "combine":
            total, weighted_losses = self.loss_combiner(losses)
            return {
                "loss": total,
                "weighted_losses": weighted_losses,
            }
        raise ValueError(f"Unsupported objective: {objective}")


class JointPretrainingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._component_sums = {
            "ntp_loss": 0.0,
            "task_loss": 0.0,
            "metric_loss": 0.0,
        }
        self._component_count = 0

    def create_scheduler(
        self, num_training_steps: int, optimizer=None
    ):
        if self.lr_scheduler is None and str(self.args.lr_scheduler_type) == "cosine":
            optimizer = self.optimizer if optimizer is None else optimizer
            warmup_steps = self.args.get_warmup_steps(num_training_steps)
            min_lr_ratio = float(self.args.min_lr_ratio)

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda
            )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def _forward_objectives(self, model, inputs):
        combined = model(objective="joint", inputs=inputs)
        return (
            combined,
            combined["ntp_output"],
            combined["task_output"],
            combined["metric_output"],
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = self._forward_objectives(model, inputs)
        combined, ntp_output, task_output, metric_output = outputs
        self._component_sums["ntp_loss"] += ntp_output.loss.detach().float().item()
        self._component_sums["task_loss"] += (
            task_output["loss"].detach().float().item()
        )
        self._component_sums["metric_loss"] += (
            metric_output["loss"].detach().float().item()
        )
        self._component_count += 1
        return (combined["loss"], combined) if return_outputs else combined["loss"]

    def log(self, logs, start_time=None):
        if self._component_count > 0:
            logs = dict(logs)
            for name, total in self._component_sums.items():
                logs[name] = total / self._component_count
            self._component_sums = {name: 0.0 for name in self._component_sums}
            self._component_count = 0
        super().log(logs, start_time=start_time)

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        dataloader = self.get_eval_dataloader(
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        self.model.eval()
        totals = torch.zeros(8, dtype=torch.float64, device=self.args.device)
        task_logits = []
        task_labels = []

        for inputs in tqdm(
            dataloader,
            desc=f"Eval step {self.state.global_step}",
            disable=not pml.is_rank0(),
            dynamic_ncols=True,
            leave=False,
        ):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                combined, ntp_output, task_output, metric_output = (
                    self._forward_objectives(self.model, inputs)
                )
            totals[0] += combined["loss"].double()
            totals[1] += ntp_output.loss.double()
            totals[2] += task_output["loss"].double()
            totals[3] += metric_output["loss_sum"].double()
            totals[4] += metric_output["abs_error_sum"].double()
            totals[5] += metric_output["squared_error_sum"].double()
            totals[6] += metric_output["pair_count"].double()
            totals[7] += 1

            gathered_logits, gathered_labels = (
                self.accelerator.gather_for_metrics(
                    (
                        task_output["logits"].detach(),
                        task_output["labels"].detach(),
                    )
                )
            )
            task_logits.append(gathered_logits.float().cpu())
            task_labels.append(gathered_labels.float().cpu())

        if pml.is_distributed():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        batch_count = max(float(totals[7].item()), 1.0)
        pair_count = max(float(totals[6].item()), 1.0)
        metrics = {
            f"{metric_key_prefix}_loss": float(totals[0].item() / batch_count),
            f"{metric_key_prefix}_ntp_loss": float(
                totals[1].item() / batch_count
            ),
            f"{metric_key_prefix}_task_loss": float(
                totals[2].item() / batch_count
            ),
            f"{metric_key_prefix}_metric_loss": float(
                totals[3].item() / pair_count
            ),
            f"{metric_key_prefix}_metric_mae": float(
                totals[4].item() / pair_count
            ),
            f"{metric_key_prefix}_metric_rmse": float(
                math.sqrt(totals[5].item() / pair_count)
            ),
            f"{metric_key_prefix}_metric_pair_count": float(totals[6].item()),
        }
        if task_logits:
            classification = compute_classification_metrics(
                EvalPrediction(
                    predictions=torch.cat(task_logits).numpy(),
                    label_ids=torch.cat(task_labels).numpy(),
                )
            )
            metrics.update(
                {
                    f"{metric_key_prefix}_task_{name}": value
                    for name, value in classification.items()
                }
            )

        if pml.is_rank0():
            print(
                f"[Eval] step={self.state.global_step} "
                + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            )
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self.model.train()
        return metrics


def embedding_cache_paths(data_args):
    paths = []
    for dataset_name in data_args.dataset:
        if dataset_name == "mimic_iv":
            paths.extend(data_args.table_text_embedding)
        elif dataset_name == "eicu":
            paths.extend(data_args.eicu_table_text_embedding)
        elif dataset_name == "ehrshot":
            paths.extend(data_args.ehrshot_table_text_embedding)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return paths


def split_ntp_paths(data_args, dataset_name, split):
    if dataset_name == "mimic_iv":
        return (
            data_args.ntp_train_info_path
            if split == "train"
            else data_args.ntp_val_info_path
        )
    if dataset_name == "eicu":
        return (
            data_args.eicu_ntp_train_info_path
            if split == "train"
            else data_args.eicu_ntp_val_info_path
        )
    if dataset_name == "ehrshot":
        return (
            data_args.ehrshot_ntp_train_info_path
            if split == "train"
            else data_args.ehrshot_ntp_val_info_path
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_ntp_dataset(data_args, split):
    datasets = []
    for dataset_name in data_args.dataset:
        processed_dir = (
            data_args.eicu_processed_dir if dataset_name == "eicu" else None
        )
        root_dir = {
            "mimic_iv": data_args.root_dir,
            "eicu": data_args.eicu_root_dir,
            "ehrshot": data_args.ehrshot_root_dir,
        }[dataset_name]
        dataset = ntp.build_dataset(
            dataset_name=dataset_name,
            root_dir=root_dir,
            processed_dir=processed_dir,
            sample_info_paths=split_ntp_paths(
                data_args, dataset_name, split
            ),
            max_samples=None,
            min_table_rows=data_args.min_table_rows,
            shuffle=split == "train",
        )
        datasets.append(dataset)
    mixed = ConcatDataset(datasets)
    limit = (
        data_args.max_ntp_train_samples
        if split == "train"
        else data_args.max_ntp_eval_samples
    )
    return Subset(mixed, range(min(limit, len(mixed)))) if limit else mixed


def build_task_dataset(data_args, split):
    task_info = tqc.get_task_info()
    binary_tasks = tqc.binary_task_names(task_info)
    parts = []
    if "mimic_iv" in data_args.dataset:
        path = (
            data_args.task_train_sample_info_path
            if split == "train"
            else data_args.task_val_sample_info_path
        )
        parts.extend(
            tqc.build_mimic_datasets(
                data_args.root_dir, tqc.resolve_sample_info_paths(path)
            )
        )
    if "eicu" in data_args.dataset:
        path = (
            data_args.eicu_task_train_sample_info_path
            if split == "train"
            else data_args.eicu_task_val_sample_info_path
        )
        tasks = [
            name
            for name in binary_tasks
            if name in tqc.get_eicu_task_info()
        ]
        parts.extend(
            tqc.build_eicu_datasets(
                data_args.eicu_root_dir,
                data_args.eicu_processed_dir,
                tqc.load_json_records(path),
                tasks,
            )
        )
    if "ehrshot" in data_args.dataset:
        path = (
            data_args.ehrshot_task_train_sample_info_path
            if split == "train"
            else data_args.ehrshot_task_val_sample_info_path
        )
        tasks = [
            name
            for name in binary_tasks
            if name in tqc.get_ehrshot_task_info()
        ]
        parts.extend(
            tqc.build_ehrshot_datasets(
                data_args.ehrshot_root_dir,
                tqc.load_csv_records(path),
                tasks,
            )
        )
    limit = (
        data_args.max_task_train_samples
        if split == "train"
        else data_args.max_task_eval_samples
    )
    return tqc.TaskQueryDataset(parts, limit)


def limit_dataset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def main():
    parser = HfArgumentParser((DataArguments, PretrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    set_seed(training_args.seed)

    text_dim, text_to_idx, embedding_matrix = pml.load_table_embeddings(
        embedding_cache_paths(data_args)
    )
    type_vocab = pml.load_type_vocab(data_args.type_vocab_file)

    ntp_train = build_ntp_dataset(data_args, "train")
    ntp_val = build_ntp_dataset(data_args, "val")
    task_train = build_task_dataset(data_args, "train")
    task_val = build_task_dataset(data_args, "val")
    task_names = sorted(
        set(task_train.task_names()) | set(task_val.task_names())
    )
    task_info = tqc.get_task_info()
    task_query_embeddings = pml.build_knowledge_query_embeddings(
        query_texts={
            task_name: task_info[task_name]["instruction"]
            for task_name in task_names
        },
        cache_path=data_args.task_query_embedding_cache,
        model_path=data_args.knowledge_encoder_path,
        base_model_path=data_args.knowledge_encoder_base_model_path,
        max_length=data_args.query_max_length,
        batch_size=data_args.query_embedding_batch_size,
    )

    phenotype_specs = pml.load_query_specs(data_args.phenotype_spec_path)
    phenotype_query_texts = {
        spec.key: spec.query_text for spec in phenotype_specs
    }
    phenotype_query_embeddings = pml.build_knowledge_query_embeddings(
        query_texts=phenotype_query_texts,
        cache_path=data_args.phenotype_query_embedding_cache,
        model_path=data_args.knowledge_encoder_path,
        base_model_path=data_args.knowledge_encoder_base_model_path,
        max_length=data_args.query_max_length,
        batch_size=data_args.query_embedding_batch_size,
    )
    phenotype_query_embedding_matrix = torch.stack(
        [phenotype_query_embeddings[spec.key] for spec in phenotype_specs],
        dim=0,
    )
    phenotype_scales = torch.tensor(
        [
            (
                float(spec.scale)
                if spec.scale is not None and float(spec.scale) > 0
                else 0.0
            )
            for spec in phenotype_specs
        ],
        dtype=torch.float,
    )
    metric_train = limit_dataset(
        pml.PreprocessedPhenotypeMetricDataset(
            cache_root=data_args.phenotype_preprocessed_input_dir,
            split="train",
            query_specs=phenotype_specs,
            text_to_idx=text_to_idx,
        ),
        data_args.max_metric_train_samples,
    )
    metric_val = limit_dataset(
        pml.PreprocessedPhenotypeMetricDataset(
            cache_root=data_args.phenotype_preprocessed_input_dir,
            split="val",
            query_specs=phenotype_specs,
            text_to_idx=text_to_idx,
        ),
        data_args.max_metric_eval_samples,
    )

    task_query_dim = int(next(iter(task_query_embeddings.values())).numel())
    phenotype_query_dim = int(phenotype_query_embedding_matrix.size(-1))
    if phenotype_query_dim != task_query_dim:
        raise ValueError(
            "Task and phenotype query embedding dimensions must match for "
            f"joint pretraining: task={task_query_dim}, "
            f"phenotype={phenotype_query_dim}"
        )
    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=max(type_vocab.values()) + 1,
        max_table_len=data_args.max_table_len,
        dim_out=task_query_dim,
    )
    model = JointPretrainingModel(
        config=config,
        embedding_matrix=embedding_matrix,
        phenotype_query_embedding_matrix=phenotype_query_embedding_matrix,
        phenotype_scales=phenotype_scales,
        training_args=training_args,
    )

    ntp_collator = ntp.NextTokenDataCollator(
        text_to_idx=text_to_idx,
        pad_idx=0,
        type_vocab=type_vocab,
        max_table_len=data_args.max_table_len,
        min_table_rows=data_args.min_table_rows,
    )
    task_collator = tqc.TaskQueryCollator(
        text_to_idx=text_to_idx,
        pad_idx=0,
        type_vocab=type_vocab,
        query_embeddings=task_query_embeddings,
        max_table_len=data_args.max_table_len,
        min_table_rows=data_args.min_table_rows,
    )
    metric_collator = pml.PreprocessedPhenotypeMetricCollator(
        max_table_len=data_args.max_table_len,
        min_table_rows=data_args.min_table_rows,
    )
    collator = MultiTaskCollator(
        ntp_collator=ntp_collator,
        task_collator=task_collator,
        metric_collator=metric_collator,
    )
    train_dataset = CycledMultiTaskDataset(
        ntp_train, task_train, metric_train
    )
    eval_dataset = CycledMultiTaskDataset(ntp_val, task_val, metric_val)

    print(f"NTP train/val: {len(ntp_train)}/{len(ntp_val)}")
    print(f"Task train/val: {len(task_train)}/{len(task_val)}")
    print(f"Metric train/val: {len(metric_train)}/{len(metric_val)}")
    print(f"Joint train/val steps: {len(train_dataset)}/{len(eval_dataset)}")
    print(f"Knowledge query dimension: {task_query_dim}")
    print(f"Phenotype metric queries: {len(phenotype_specs)}")

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience
        )
    ]
    trainer = JointPretrainingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train(
        resume_from_checkpoint=getattr(
            training_args, "resume_from_checkpoint", None
        )
    )
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
