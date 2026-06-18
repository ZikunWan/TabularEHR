import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from . import phenotype_metric_learning as pml
except ImportError:
    import phenotype_metric_learning as pml


@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    output_dir: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/phenotype_triplet_learning"
    )
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
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
    wandb_project: Optional[str] = field(default="Phenotype_Triplet_Learning")
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    triplet_alpha: float = field(default=0.1)
    positive_k: int = field(default=3)
    min_clinical_gap: float = field(default=0.1)
    min_shared_phenotypes: int = field(default=3)
    covariance_shrinkage: float = field(default=0.1)
    covariance_max_samples: int = field(default=50000)
    covariance_rank: int = field(default=64)
    clinical_reference_path: Optional[str] = field(default=None)
    min_lr_ratio: float = field(default=0.1)

    def __post_init__(self):
        super().__post_init__()
        if self.triplet_alpha < 0:
            raise ValueError("--triplet_alpha must be non-negative.")
        if self.positive_k <= 0:
            raise ValueError("--positive_k must be positive.")
        if self.min_clinical_gap < 0:
            raise ValueError("--min_clinical_gap must be non-negative.")
        if self.min_shared_phenotypes <= 0:
            raise ValueError("--min_shared_phenotypes must be positive.")
        if not 0 <= self.covariance_shrinkage <= 1:
            raise ValueError("--covariance_shrinkage must be in [0, 1].")
        if self.covariance_max_samples <= 1:
            raise ValueError("--covariance_max_samples must be greater than 1.")
        if self.covariance_rank <= 0:
            raise ValueError("--covariance_rank must be positive.")
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        self.eval_strategy = "steps"
        self.load_best_model_at_end = True


def _collect_phenotype_rows(
    dataset: Dataset,
    query_specs: Sequence[pml.PhenotypeQuerySpec],
    feature_indices: torch.Tensor,
    max_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    extractor = pml.PhenotypeValueExtractor(list(query_specs))
    if len(dataset) <= max_samples:
        indices = range(len(dataset))
    else:
        step = len(dataset) / float(max_samples)
        indices = (min(int(i * step), len(dataset) - 1) for i in range(max_samples))

    value_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []
    for index in indices:
        sample = dataset[index]
        if "phenotype_values" in sample:
            values = sample["phenotype_values"].float().index_select(
                0, feature_indices
            )
            mask = sample["phenotype_mask"].bool().index_select(
                0, feature_indices
            )
        else:
            raw_values, raw_mask = extractor(sample["table"])
            values = torch.tensor(raw_values, dtype=torch.float).index_select(
                0, feature_indices
            )
            mask = torch.tensor(raw_mask, dtype=torch.bool).index_select(
                0, feature_indices
            )
        if mask.any():
            value_rows.append(values)
            mask_rows.append(mask)

    if len(value_rows) < 2:
        raise ValueError("At least two phenotype-bearing training samples are required.")
    return torch.stack(value_rows), torch.stack(mask_rows)


def _parse_reference_range(text: str) -> Optional[Tuple[float, float]]:
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.search(
        rf"({number})\s*(?:-|–|—|~|to)\s*({number})",
        str(text),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    low, high = map(float, match.groups())
    return (low, high) if high > low else None


def _normalize_reference_item(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _normalize_reference_unit(text: str) -> str:
    unit = str(text).strip().lower()
    aliases = {
        "mg/dl.": "mg/dl",
        "mm hg": "mmhg",
        "mm[hg]": "mmhg",
        "iu/l": "u/l",
        "units/l": "u/l",
        "unit/l": "u/l",
        "seconds": "sec",
        "second": "sec",
        "thousand/ul": "k/ul",
        "thousands/ul": "k/ul",
        "x10e3/ul": "k/ul",
        "10^3/ul": "k/ul",
        "10e3/ul": "k/ul",
        "10^6/ul": "m/ul",
        "million/ul": "m/ul",
        "mil/ul": "m/ul",
        "beats per minute": "bpm",
        "[in_us]": "inch",
    }
    return aliases.get(unit, unit)


def _reference_lookup_keys(item: str, unit: str, key: Optional[str] = None) -> List[str]:
    item = str(item).strip()
    unit = str(unit).strip()
    keys = []
    if key:
        keys.append(str(key))
    keys.append(f"{item.lower()}|{unit.lower()}")
    keys.append(
        f"{_normalize_reference_item(item)}|{_normalize_reference_unit(unit)}"
    )
    return keys


def _add_reference_range(
    ranges: dict,
    item: str,
    unit: str,
    low: float,
    high: float,
    key: Optional[str] = None,
) -> None:
    if high <= low:
        raise ValueError(
            f"Invalid clinical reference range for {item}|{unit}: {(low, high)}"
        )
    for lookup_key in _reference_lookup_keys(item, unit, key):
        ranges[lookup_key] = (low, high)


def _load_reference_ranges(
    path: Optional[str],
) -> dict:
    if not path:
        return {}
    ranges = {}
    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                item = row.get("item") or row.get("concept") or row.get("name")
                unit = row.get("unit", "")
                low = row.get("ref_low") or row.get("low")
                high = row.get("ref_high") or row.get("high")
                if not item or low is None or high is None:
                    continue
                _add_reference_range(
                    ranges=ranges,
                    item=str(item),
                    unit=str(unit),
                    low=float(low),
                    high=float(high),
                    key=row.get("key"),
                )
    else:
        with open(path, "r", encoding="utf-8") as file:
            raw_ranges = json.load(file)
        for key, raw_value in raw_ranges.items():
            if isinstance(raw_value, dict):
                low = raw_value.get("low", raw_value.get("ref_low"))
                high = raw_value.get("high", raw_value.get("ref_high"))
                item = raw_value.get("item", key.split("|", 1)[0])
                unit = raw_value.get("unit", key.split("|", 1)[1] if "|" in key else "")
            else:
                low, high = raw_value
                item = key.split("|", 1)[0]
                unit = key.split("|", 1)[1] if "|" in key else ""
            _add_reference_range(
                ranges=ranges,
                item=str(item),
                unit=str(unit),
                low=float(low),
                high=float(high),
                key=str(key),
            )
    return ranges


def _get_external_reference_range(
    external_ranges: dict,
    spec: pml.PhenotypeQuerySpec,
) -> Optional[Tuple[float, float]]:
    for lookup_key in _reference_lookup_keys(spec.item, spec.unit, spec.key):
        if lookup_key in external_ranges:
            return external_ranges[lookup_key]
    return None


def select_clinical_feature_indices(
    query_specs: Sequence[pml.PhenotypeQuerySpec],
) -> torch.Tensor:
    def priority(spec: pml.PhenotypeQuerySpec) -> Tuple[int, int, str]:
        window = spec.window_name.strip().lower()
        statistic = spec.statistic.strip().lower()
        window_priority = 0 if window in {"full", "full encounter"} else 1
        statistic_priority = {
            "latest": 0,
            "mean": 1,
            "first": 2,
            "max": 3,
            "min": 4,
        }.get(statistic, 5)
        return window_priority, statistic_priority, spec.key

    selected = {}
    for index, spec in enumerate(query_specs):
        group = (spec.item.strip().lower(), spec.unit.strip().lower())
        candidate = (priority(spec), index)
        if group not in selected or candidate[0] < selected[group][0]:
            selected[group] = candidate
    indices = sorted(candidate[1] for candidate in selected.values())
    if not indices:
        raise ValueError("No clinical phenotype features were selected.")
    return torch.tensor(indices, dtype=torch.long)


def estimate_clinical_statistics(
    dataset: Dataset,
    query_specs: Sequence[pml.PhenotypeQuerySpec],
    shrinkage: float,
    max_samples: int,
    covariance_rank: int,
    reference_path: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_indices = select_clinical_feature_indices(query_specs)
    selected_specs = [query_specs[index] for index in feature_indices.tolist()]
    external_ranges = _load_reference_ranges(reference_path)
    if external_ranges:
        has_reference = torch.tensor(
            [
                _get_external_reference_range(external_ranges, spec) is not None
                for spec in selected_specs
            ],
            dtype=torch.bool,
        )
        if not has_reference.any():
            raise ValueError(
                "No selected clinical phenotype matches the external reference "
                f"file: {reference_path}"
            )
        feature_indices = feature_indices[has_reference]
        selected_specs = [
            spec for spec, keep in zip(selected_specs, has_reference.tolist())
            if keep
        ]
    values, mask = _collect_phenotype_rows(
        dataset, query_specs, feature_indices, max_samples
    )
    observed_count = mask.sum(dim=0)
    usable = observed_count >= 2
    if not usable.any():
        raise ValueError(
            "No selected clinical phenotype has at least two observations."
        )
    feature_indices = feature_indices[usable]
    values = values[:, usable]
    mask = mask[:, usable]
    selected_specs = [
        spec for spec, keep in zip(selected_specs, usable.tolist()) if keep
    ]
    count = mask.float().sum(dim=0)
    empirical_center = values.masked_fill(~mask, 0.0).sum(dim=0) / count
    centered = (values - empirical_center).masked_fill(~mask, 0.0)
    empirical_scale = torch.sqrt(
        centered.pow(2).sum(dim=0) / count
    ).clamp_min(1e-6)

    reference_center = empirical_center.clone()
    reference_scale = empirical_scale.clone()
    reference_count = 0
    for index, spec in enumerate(selected_specs):
        reference_range = _get_external_reference_range(external_ranges, spec)
        if reference_range is None and not external_ranges:
            reference_range = _parse_reference_range(spec.normal_range)
        if reference_range is not None:
            low, high = reference_range
            reference_center[index] = (low + high) / 2.0
            reference_scale[index] = (high - low) / 2.0
            reference_count += 1
        elif (
            not external_ranges
            and spec.mean is not None
            and spec.scale is not None
            and float(spec.scale) > 0
        ):
            reference_center[index] = float(spec.mean)
            reference_scale[index] = float(spec.scale)

    normalized = (values - reference_center) / reference_scale
    normalized_mean = (
        normalized.masked_fill(~mask, 0.0).sum(dim=0) / count
    )
    filled = torch.where(mask, normalized, normalized_mean)
    filled = filled - filled.mean(dim=0, keepdim=True)
    covariance = filled.transpose(0, 1) @ filled
    covariance = covariance / max(filled.size(0) - 1, 1)
    diagonal = covariance.diagonal().clamp_min(1e-4)
    covariance = (
        (1.0 - shrinkage) * covariance
        + shrinkage * torch.diag(diagonal)
        + 1e-5 * torch.eye(covariance.size(0))
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
    rank = min(int(covariance_rank), covariance.size(0))
    retained_eigenvalues = eigenvalues[-rank:]
    retained_eigenvectors = eigenvectors[:, -rank:]
    precision_factor = retained_eigenvectors / torch.sqrt(
        retained_eigenvalues.clamp_min(1e-5)
    ).unsqueeze(0)
    if pml.is_rank0():
        print(
            "Clinical statistics: "
            f"samples={filled.size(0)}, selected_features={len(selected_specs)}/"
            f"{len(query_specs)}, precision_rank={rank}, "
            f"explicit_reference_ranges={reference_count}/{len(selected_specs)}, "
            f"shrinkage={shrinkage:.3f}, "
            f"retained_eigenvalue_range="
            f"[{retained_eigenvalues.min().item():.4g}, "
            f"{retained_eigenvalues.max().item():.4g}]"
        )
        if reference_count < len(selected_specs):
            print(
                "Warning: phenotypes without a parseable two-sided normal_range "
                "use training-set mean/std normalization."
            )
    return (
        feature_indices,
        reference_center.float(),
        reference_scale.float(),
        precision_factor.float(),
    )


class PhenotypeTripletModel(PreTrainedModel):
    config_class = pml.LongTableEncoder1DConfig
    base_model_prefix = "encoder"

    def __init__(
        self,
        config,
        embedding_matrix: torch.Tensor,
        clinical_feature_indices: torch.Tensor,
        clinical_center: torch.Tensor,
        clinical_scale: torch.Tensor,
        clinical_precision_factor: torch.Tensor,
        triplet_alpha: float,
        positive_k: int,
        min_clinical_gap: float,
        min_shared_phenotypes: int,
    ):
        super().__init__(config)
        self.encoder = pml.LongTableEncoder1D(config)
        self.adapter = pml.QFormerAdapter(config)
        hidden_size = config.dim_out if config.dim_out is not None else config.dim
        self.pooling = pml.AttentionPooling(hidden_size)
        self.text_embedding_matrix = embedding_matrix.cpu()
        self.register_buffer(
            "clinical_feature_indices", clinical_feature_indices.long()
        )
        self.register_buffer("clinical_center", clinical_center.float())
        self.register_buffer("clinical_scale", clinical_scale.float())
        self.register_buffer(
            "clinical_precision_factor", clinical_precision_factor.float()
        )
        self.triplet_alpha = float(triplet_alpha)
        self.positive_k = int(positive_k)
        self.min_clinical_gap = float(min_clinical_gap)
        self.min_shared_phenotypes = int(min_shared_phenotypes)
        self.post_init()

    def text_lookup(
        self, token_ids: torch.Tensor, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        flat = self.text_embedding_matrix.index_select(
            0, token_ids.reshape(-1).cpu()
        )
        flat = flat.to(device=device, dtype=dtype, non_blocking=True)
        return flat.view(*token_ids.shape, flat.size(-1))

    def encode_table(
        self,
        item_ids,
        unit_ids,
        value_text_ids,
        times,
        numeric_values,
        numeric_mask,
        seq_mask,
        type_ids,
    ):
        dtype = self.encoder.embedding.item_proj.weight.dtype
        device = self.encoder.embedding.item_proj.weight.device
        hidden_states, hidden_mask = self.encoder(
            item_emb=self.text_lookup(item_ids, dtype, device),
            unit_emb=self.text_lookup(unit_ids, dtype, device),
            value_emb=self.text_lookup(value_text_ids, dtype, device),
            times=times,
            numeric_values=numeric_values,
            numeric_mask=numeric_mask,
            seq_mask=seq_mask,
            type_ids=type_ids,
            return_mask=True,
        )
        hidden_states = self.adapter(hidden_states, hidden_mask)
        pooled_mask = torch.ones(
            hidden_states.shape[:2],
            dtype=hidden_mask.dtype,
            device=hidden_mask.device,
        )
        return self.pooling(hidden_states, pooled_mask)

    @staticmethod
    def _clinical_distance_matrix(
        local_values: torch.Tensor,
        global_values: torch.Tensor,
        local_mask: torch.Tensor,
        global_mask: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        precision_factor: torch.Tensor,
        min_shared: int,
    ) -> torch.Tensor:
        local_values = local_values.float()
        global_values = global_values.float()
        center = center.float()
        scale = scale.float()
        precision_factor = precision_factor.float()
        local_normalized = (local_values - center) / scale
        global_normalized = (global_values - center) / scale
        shared = local_mask.unsqueeze(1) & global_mask.unsqueeze(0)
        delta = global_normalized.unsqueeze(0) - local_normalized.unsqueeze(1)
        delta = delta.masked_fill(~shared, 0.0)
        whitened_delta = delta @ precision_factor
        shared_count = shared.sum(dim=-1)
        correction = local_values.size(-1) / shared_count.clamp_min(1)
        distances = torch.sqrt(
            (whitened_delta.pow(2).sum(dim=-1) * correction).clamp_min(0.0)
            + 1e-12
        )
        return distances.masked_fill(shared_count < min_shared, float("inf"))

    @staticmethod
    def _mine_semi_hard_triplets(
        clinical_distances: torch.Tensor,
        embedding_distances: torch.Tensor,
        self_indices: torch.Tensor,
        positive_k: int,
        min_clinical_gap: float,
        alpha: float,
    ):
        clinical = clinical_distances.clone()
        clinical.scatter_(1, self_indices.view(-1, 1), float("inf"))
        k = min(positive_k, clinical.size(1))
        positive_clinical, positive_candidates = clinical.topk(
            k, dim=1, largest=False
        )
        valid_positive = torch.isfinite(positive_clinical)
        candidate_embedding = embedding_distances.gather(1, positive_candidates)
        candidate_embedding = candidate_embedding.masked_fill(
            ~valid_positive, float("-inf")
        )
        positive_slot = candidate_embedding.argmax(dim=1)
        positive_indices = positive_candidates.gather(
            1, positive_slot.view(-1, 1)
        ).squeeze(1)
        anchor_indices = torch.arange(
            clinical.size(0), device=clinical.device
        )
        d_clin_ap = clinical[anchor_indices, positive_indices]
        d_emb_ap = embedding_distances[anchor_indices, positive_indices]

        positive_pool = torch.zeros_like(clinical, dtype=torch.bool)
        positive_pool.scatter_(1, positive_candidates, valid_positive)
        negative_candidates = (
            torch.isfinite(clinical)
            & ~positive_pool
            & (clinical >= d_clin_ap.unsqueeze(1) + min_clinical_gap)
        )
        margins = alpha * (clinical - d_clin_ap.unsqueeze(1))
        semi_hard = (
            negative_candidates
            & (embedding_distances > d_emb_ap.unsqueeze(1))
            & (
                embedding_distances
                < d_emb_ap.unsqueeze(1) + margins
            )
        )
        semi_hard_distances = embedding_distances.masked_fill(
            ~semi_hard, float("inf")
        )
        semi_hard_indices = semi_hard_distances.argmin(dim=1)
        has_semi_hard = semi_hard.any(dim=1)

        fallback_distances = embedding_distances.masked_fill(
            ~negative_candidates, float("inf")
        )
        fallback_indices = fallback_distances.argmin(dim=1)
        negative_indices = torch.where(
            has_semi_hard, semi_hard_indices, fallback_indices
        )
        valid = (
            valid_positive.any(dim=1)
            & negative_candidates.any(dim=1)
        )
        return positive_indices, negative_indices, valid, has_semi_hard & valid

    @staticmethod
    def _soft_triplet_terms(
        d_emb_ap: torch.Tensor,
        d_emb_an: torch.Tensor,
        d_clin_ap: torch.Tensor,
        d_clin_an: torch.Tensor,
        alpha: float,
    ):
        clinical_gap = d_clin_an - d_clin_ap
        dynamic_margin = alpha * clinical_gap
        terms = F.relu(d_emb_ap - d_emb_an + dynamic_margin)
        return terms, dynamic_margin, clinical_gap

    def forward(self, phenotype_values, phenotype_mask, labels=None, **table_inputs):
        local_embeddings = F.normalize(self.encode_table(**table_inputs), dim=-1)
        global_embeddings = pml.all_gather_with_grad(local_embeddings)
        feature_indices = self.clinical_feature_indices.to(
            phenotype_values.device
        )
        phenotype_values = phenotype_values.index_select(1, feature_indices)
        phenotype_mask = phenotype_mask.index_select(1, feature_indices)
        global_values = pml.all_gather_tensor(
            phenotype_values.to(local_embeddings.device, local_embeddings.dtype)
        )
        global_mask = pml.all_gather_tensor(
            phenotype_mask.to(local_embeddings.device).bool()
        )
        local_values = phenotype_values.to(
            local_embeddings.device, local_embeddings.dtype
        )
        local_mask = phenotype_mask.to(local_embeddings.device).bool()
        center = self.clinical_center.to(
            local_embeddings.device, local_embeddings.dtype
        )
        scale = self.clinical_scale.to(
            local_embeddings.device, local_embeddings.dtype
        )
        precision_factor = self.clinical_precision_factor.to(
            local_embeddings.device, local_embeddings.dtype
        )
        clinical_distances = self._clinical_distance_matrix(
            local_values=local_values,
            global_values=global_values,
            local_mask=local_mask,
            global_mask=global_mask,
            center=center,
            scale=scale,
            precision_factor=precision_factor,
            min_shared=self.min_shared_phenotypes,
        )
        embedding_distances = torch.cdist(
            local_embeddings.float(), global_embeddings.float(), p=2
        ).to(local_embeddings.dtype)
        batch_start = pml.gather_batch_start(
            local_embeddings.size(0), local_embeddings.device
        )
        self_indices = torch.arange(
            local_embeddings.size(0), device=local_embeddings.device
        ) + batch_start
        positive_indices, negative_indices, triplet_mask, semi_hard_mask = (
            self._mine_semi_hard_triplets(
                clinical_distances=clinical_distances,
                embedding_distances=embedding_distances,
                self_indices=self_indices,
                positive_k=self.positive_k,
                min_clinical_gap=self.min_clinical_gap,
                alpha=self.triplet_alpha,
            )
        )
        triplet_count = triplet_mask.float().sum()
        if triplet_count <= 0:
            zero = (local_embeddings.sum() + global_embeddings.sum()) * 0.0
            return zero, {
                "loss_sum": zero.detach(),
                "order_correct_sum": zero.detach(),
                "margin_correct_sum": zero.detach(),
                "semi_hard_sum": zero.detach(),
                "clinical_gap_sum": zero.detach(),
                "triplet_count": zero.detach(),
            }

        anchor_indices = torch.arange(
            local_embeddings.size(0), device=local_embeddings.device
        )
        d_emb_ap = embedding_distances[anchor_indices, positive_indices]
        d_emb_an = embedding_distances[anchor_indices, negative_indices]
        d_clin_ap = clinical_distances[anchor_indices, positive_indices]
        d_clin_an = clinical_distances[anchor_indices, negative_indices]
        triplet_terms, dynamic_margin, clinical_gap = self._soft_triplet_terms(
            d_emb_ap=d_emb_ap,
            d_emb_an=d_emb_an,
            d_clin_ap=d_clin_ap,
            d_clin_an=d_clin_an,
            alpha=self.triplet_alpha,
        )
        loss_sum = triplet_terms[triplet_mask].sum()
        loss = loss_sum / triplet_count
        return loss, {
            "loss_sum": loss_sum.detach(),
            "order_correct_sum": (
                (d_emb_an > d_emb_ap)[triplet_mask].float().sum()
            ).detach(),
            "margin_correct_sum": (
                (d_emb_an >= d_emb_ap + dynamic_margin)[triplet_mask]
                .float()
                .sum()
            ).detach(),
            "semi_hard_sum": semi_hard_mask.float().sum().detach(),
            "clinical_gap_sum": clinical_gap[triplet_mask].sum().detach(),
            "triplet_count": triplet_count.detach(),
        }


class PhenotypeTripletTrainer(Trainer):
    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        if self.lr_scheduler is None and str(self.args.lr_scheduler_type) == "cosine":
            optimizer = self.optimizer if optimizer is None else optimizer
            warmup_steps = self.args.get_warmup_steps(num_training_steps)
            min_lr_ratio = float(self.args.min_lr_ratio)

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda
            )
            return self.lr_scheduler
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("labels", None)
        loss, outputs = model(**inputs)
        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        dataloader = self.get_eval_dataloader(
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        self.model.eval()
        totals = torch.zeros(6, dtype=torch.float64, device=self.args.device)

        for inputs in tqdm(
            dataloader,
            desc=f"Eval step {self.state.global_step}",
            disable=not pml.is_rank0(),
            dynamic_ncols=True,
            leave=False,
        ):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                _, outputs = self.compute_loss(
                    self.model, inputs, return_outputs=True
                )
            totals[0] += outputs["loss_sum"].double()
            totals[1] += outputs["order_correct_sum"].double()
            totals[2] += outputs["margin_correct_sum"].double()
            totals[3] += outputs["semi_hard_sum"].double()
            totals[4] += outputs["clinical_gap_sum"].double()
            totals[5] += outputs["triplet_count"].double()

        if pml.is_distributed():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        triplet_count = max(float(totals[5].item()), 1.0)
        metrics = {
            f"{metric_key_prefix}_loss": float(totals[0].item() / triplet_count),
            f"{metric_key_prefix}_order_accuracy": float(
                totals[1].item() / triplet_count
            ),
            f"{metric_key_prefix}_triplet_accuracy": float(
                totals[2].item() / triplet_count
            ),
            f"{metric_key_prefix}_semi_hard_fraction": float(
                totals[3].item() / triplet_count
            ),
            f"{metric_key_prefix}_clinical_gap": float(
                totals[4].item() / triplet_count
            ),
            f"{metric_key_prefix}_triplet_count": float(totals[5].item()),
        }
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


def main():
    parser = HfArgumentParser((pml.DataArguments, TrainingArgumentsCustom))
    data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    set_seed(training_args.seed)

    print("Phenotype clinical-distance triplet learning")
    print(f"Datasets: {', '.join(data_args.dataset)}")
    print(f"Pretrained path: {data_args.pretrained_path}")

    if data_args.preprocessed_inputs_only:
        if not data_args.phenotype_spec_path:
            raise ValueError(
                "--phenotype_spec_path is required with "
                "--preprocessed_inputs_only true."
            )
        train_records = train_datasets = val_records = val_datasets = None
        query_specs = pml.load_query_specs(data_args.phenotype_spec_path)
    else:
        train_records, train_datasets = pml.build_split_records(data_args, "train")
        val_records, val_datasets = pml.build_split_records(data_args, "val")
        query_specs = pml.build_query_specs(
            data_args, train_records, train_datasets
        )
    print(f"Loaded phenotype query specs: {len(query_specs)}")

    text_dim, text_to_idx, embedding_matrix = pml.load_table_embeddings(
        pml.get_embedding_cache_paths(data_args)
    )
    type_vocab = pml.load_type_vocab(data_args.type_vocab_file)
    config = pml.LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=max(type_vocab.values()) + 1,
        max_table_len=data_args.max_table_len,
    )

    if data_args.preprocessed_inputs_only:
        train_dataset = pml.PreprocessedPhenotypeMetricDataset(
            cache_root=data_args.preprocessed_input_dir,
            split="train",
            query_specs=query_specs,
            text_to_idx=text_to_idx,
        )
        val_dataset = pml.PreprocessedPhenotypeMetricDataset(
            cache_root=data_args.preprocessed_input_dir,
            split="val",
            query_specs=query_specs,
            text_to_idx=text_to_idx,
        )
        collator = pml.PreprocessedPhenotypeMetricCollator(
            max_table_len=data_args.max_table_len,
            min_table_rows=data_args.min_table_rows,
        )
    else:
        train_dataset = pml.PhenotypeMetricDataset(
            records=train_records,
            datasets=train_datasets,
            max_table_len=data_args.max_table_len,
            is_eval=False,
        )
        val_dataset = pml.PhenotypeMetricDataset(
            records=val_records,
            datasets=val_datasets,
            max_table_len=data_args.max_table_len,
            is_eval=True,
        )
        collator = pml.PhenotypeMetricCollator(
            text_to_idx=text_to_idx,
            type_vocab=type_vocab,
            query_specs=query_specs,
            max_table_len=data_args.max_table_len,
            min_table_rows=data_args.min_table_rows,
            augmentation_seed=training_args.seed,
        )

    (
        clinical_feature_indices,
        clinical_center,
        clinical_scale,
        clinical_precision_factor,
    ) = (
        estimate_clinical_statistics(
            dataset=train_dataset,
            query_specs=query_specs,
            shrinkage=training_args.covariance_shrinkage,
            max_samples=training_args.covariance_max_samples,
            covariance_rank=training_args.covariance_rank,
            reference_path=training_args.clinical_reference_path,
        )
    )
    model = PhenotypeTripletModel(
        config=config,
        embedding_matrix=embedding_matrix,
        clinical_feature_indices=clinical_feature_indices,
        clinical_center=clinical_center,
        clinical_scale=clinical_scale,
        clinical_precision_factor=clinical_precision_factor,
        triplet_alpha=training_args.triplet_alpha,
        positive_k=training_args.positive_k,
        min_clinical_gap=training_args.min_clinical_gap,
        min_shared_phenotypes=training_args.min_shared_phenotypes,
    )
    model = pml.load_matching_weights(model, data_args.pretrained_path)

    trainer = PhenotypeTripletTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train(
        resume_from_checkpoint=getattr(
            training_args, "resume_from_checkpoint", None
        )
    )
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
