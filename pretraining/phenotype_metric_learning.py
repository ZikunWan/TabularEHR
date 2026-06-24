import builtins
import bisect
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from torch.utils.data import Dataset, get_worker_info
from tqdm.auto import tqdm
from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.mimic.mimic_dataset import MIMICIV
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.text_encoder import TextEncoder
from models.phenotype_metric_model import (
    AttentionPooling,
    PhenotypeMetricModel,
    all_gather_tensor,
    all_gather_with_grad,
    gather_batch_start,
    is_distributed,
)
from utils.collate import build_table_token_tensors
from utils.load_embedding import build_text_to_idx


def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0
    local_rank = os.environ.get("LOCAL_RANK")
    return local_rank is None or int(local_rank) in (-1, 0)


def _configure_non_main_process_logging() -> None:
    if _is_main_process():
        return
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("ACCELERATE_LOG_LEVEL", "error")
    logging.basicConfig(level=logging.ERROR, force=True)
    for logger_name in ("transformers", "accelerate", "deepspeed", "torch", "torch.distributed"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


_configure_non_main_process_logging()


def is_rank0() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    rank = os.environ.get("RANK")
    return rank is None or int(rank) == 0


def rank0_print(*args, **kwargs) -> None:
    if is_rank0():
        builtins.print(*args, **kwargs)


print = rank0_print


def stable_seed(text: str, base_seed: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def normalize_table(table: Optional[pd.DataFrame], max_table_len: Optional[int]) -> Optional[pd.DataFrame]:
    if table is None or table.empty:
        return None
    table = table.copy()
    for column in ["Time", "Item", "Value", "Unit", "Category"]:
        if column not in table.columns:
            table[column] = ""
    table = table[["Time", "Item", "Value", "Unit", "Category"]]
    table["Time"] = pd.to_datetime(table["Time"], errors="coerce")
    table = table.sort_values("Time").reset_index(drop=True)
    if max_table_len is not None:
        table = table.tail(max_table_len).reset_index(drop=True)
    return table


def load_type_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return {str(key): int(value) for key, value in json.load(f).items()}


def load_table_embeddings(cache_paths: List[str]):
    embedding_cache = {}
    text_dim = None
    for cache_path in cache_paths:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        embedding_cache.update(data["embeddings"])
        text_dim = int(data["text_dim"])
        print(f"Loaded {len(data['embeddings'])} table embeddings from {cache_path}")
    if text_dim is None:
        raise ValueError("No table text embeddings loaded.")

    text_to_idx = build_text_to_idx(list(embedding_cache.keys()))
    embedding_matrix = torch.empty(len(text_to_idx), text_dim)
    for text, idx in text_to_idx.items():
        embedding_matrix[idx] = embedding_cache[text]
    return text_dim, text_to_idx, embedding_matrix


def load_table_text_to_idx(cache_paths: List[str]):
    vocab_keys = {}
    text_dim = None
    for cache_path in cache_paths:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        for text in data["embeddings"]:
            vocab_keys.setdefault(text, None)
        text_dim = int(data["text_dim"])
        print(f"Loaded {len(data['embeddings'])} table vocabulary entries from {cache_path}")
    if text_dim is None:
        raise ValueError("No table text embeddings loaded.")
    return text_dim, build_text_to_idx(list(vocab_keys))


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        return load_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def resolve_checkpoint_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isfile(path):
        return path
    for filename in ("model.safetensors", "pytorch_model.bin", "best.pt"):
        candidate = os.path.join(path, filename)
        if os.path.exists(candidate):
            return candidate
    return None


def load_matching_weights(model: nn.Module, pretrained_path: Optional[str]) -> nn.Module:
    checkpoint_path = resolve_checkpoint_path(pretrained_path)
    if checkpoint_path is None:
        if pretrained_path:
            print(f"Pretrained checkpoint not found, training from init: {pretrained_path}")
        return model

    state_dict = strip_module_prefix(load_checkpoint_state_dict(checkpoint_path))
    state_dict.pop("text_embedding.weight", None)
    model_state = model.state_dict()
    matched_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    missing, unexpected = model.load_state_dict(matched_state, strict=False)
    print(
        f"Loaded checkpoint: {checkpoint_path} "
        f"(matched={len(matched_state)}, missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return model


@dataclass
class PhenotypeQuerySpec:
    key: str
    item: str
    query_text: str
    aliases: List[str] = field(default_factory=list)
    statistic: str = "latest"
    unit: str = ""
    description: str = ""
    normal_range: str = ""
    window_name: str = "full encounter"
    window_start_hours: Optional[float] = None
    window_end_hours: Optional[float] = None
    category_regex: str = "^measurement$"
    item_regex: Optional[str] = None
    transform: str = "none"
    mean: Optional[float] = None
    scale: Optional[float] = None


PREPROCESSED_INPUT_FORMAT_VERSION = 1
PREPROCESSED_SEQUENCE_DTYPES = {
    "item_ids": np.int32,
    "unit_ids": np.int32,
    "value_text_ids": np.int32,
    "times": np.float32,
    "numeric_values": np.float32,
    "numeric_mask": np.uint8,
    "type_ids": np.int32,
}


def phenotype_spec_fingerprint(query_specs: List[PhenotypeQuerySpec]) -> str:
    payload = json.dumps(
        [asdict(spec) for spec in query_specs],
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_vocab_fingerprint(text_to_idx: Dict[str, int]) -> str:
    digest = hashlib.sha256()
    for text, idx in sorted(text_to_idx.items(), key=lambda item: item[1]):
        encoded = str(text).encode("utf-8")
        digest.update(int(idx).to_bytes(8, byteorder="little", signed=False))
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def sanitize_key(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return text or "phenotype"


def parse_time_window(raw: str) -> Dict[str, Optional[float]]:
    parts = str(raw).split(":")
    if len(parts) != 3:
        raise ValueError("time windows must use 'name:start_hours:end_hours', with blank start/end allowed.")
    name, start, end = parts
    return {
        "name": name or "full encounter",
        "start": float(start) if start.strip() else None,
        "end": float(end) if end.strip() else None,
    }


def build_query_text(spec: Dict[str, Any]) -> str:
    parts = [f"Continuous clinical measurement: {spec.get('item', '')}."]
    if spec.get("description"):
        parts.append(f"Clinical meaning: {spec['description']}.")
    if spec.get("unit"):
        parts.append(f"Unit: {spec['unit']}.")
    if spec.get("normal_range"):
        parts.append(f"Normal range: {spec['normal_range']}.")
    window_name = spec.get("window_name") or "full encounter"
    statistic = spec.get("statistic") or "latest"
    parts.append(f"Target: {statistic} value during {window_name}.")
    return " ".join(parts)


def make_query_spec(raw_spec: Dict[str, Any]) -> PhenotypeQuerySpec:
    spec = dict(raw_spec)
    item = str(spec.get("item") or spec.get("name") or "").strip()
    if not item:
        raise ValueError(f"Phenotype query spec is missing an item/name: {raw_spec}")
    aliases = [str(value).strip() for value in spec.get("aliases", []) if str(value).strip()]
    statistic = str(spec.get("statistic", "latest")).strip().lower()
    window_name = str(spec.get("window_name", "full encounter")).strip() or "full encounter"
    key = str(spec.get("key") or "").strip()
    if not key:
        key = sanitize_key(f"{item}_{spec.get('unit', '')}_{window_name}_{statistic}")
    spec.setdefault("query_text", build_query_text({**spec, "item": item, "statistic": statistic}))
    return PhenotypeQuerySpec(
        key=key,
        item=item,
        query_text=str(spec["query_text"]),
        aliases=aliases,
        statistic=statistic,
        unit=str(spec.get("unit", "")),
        description=str(spec.get("description", "")),
        normal_range=str(spec.get("normal_range", "")),
        window_name=window_name,
        window_start_hours=spec.get("window_start_hours"),
        window_end_hours=spec.get("window_end_hours"),
        category_regex=str(spec.get("category_regex", "^measurement$")),
        item_regex=spec.get("item_regex"),
        transform=str(spec.get("transform", "none")).lower(),
        mean=spec.get("mean"),
        scale=spec.get("scale"),
    )


def load_query_specs(path: str) -> List[PhenotypeQuerySpec]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            raw_specs = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, dict):
                raw_specs = []
                for key, value in data.items():
                    spec = dict(value)
                    spec.setdefault("key", key)
                    raw_specs.append(spec)
            else:
                raw_specs = list(data)
    specs = [make_query_spec(spec) for spec in raw_specs]
    if not specs:
        raise ValueError(f"No phenotype query specs loaded from {path}")
    return specs


def category_is_continuous(category: Any, pattern: str) -> bool:
    return re.search(pattern, str(category), flags=re.IGNORECASE) is not None


def apply_value_transform(values: pd.Series, transform: str) -> pd.Series:
    if transform == "none":
        return values
    if transform == "log1p":
        return values.where(values >= 0).map(lambda value: math.log1p(value) if pd.notna(value) else value)
    if transform == "log":
        return values.where(values > 0).map(lambda value: math.log(value) if pd.notna(value) else value)
    raise ValueError(f"Unsupported value transform: {transform}")


def aggregate_phenotype_value(
    selected: pd.DataFrame,
    spec: PhenotypeQuerySpec,
    anchor_time: Optional[pd.Timestamp],
) -> Optional[float]:
    selected = selected.copy()
    selected_values = apply_value_transform(
        pd.to_numeric(selected["Value"], errors="coerce"),
        spec.transform,
    )
    selected = selected.loc[selected_values.notna()].copy()
    selected["numeric_value"] = selected_values.loc[selected_values.notna()].astype(float)
    if selected.empty:
        return None

    if anchor_time is not None and selected["Time"].notna().any():
        selected["hours_from_anchor"] = (selected["Time"] - anchor_time).dt.total_seconds() / 3600.0
        if spec.window_start_hours is not None:
            selected = selected[selected["hours_from_anchor"] >= float(spec.window_start_hours)]
        if spec.window_end_hours is not None:
            selected = selected[selected["hours_from_anchor"] <= float(spec.window_end_hours)]
    if selected.empty:
        return None

    selected = selected.sort_values("Time").reset_index(drop=True)
    values = selected["numeric_value"].astype(float)
    statistic = spec.statistic

    if statistic in {"latest", "last"}:
        return float(values.iloc[-1])
    if statistic == "first":
        return float(values.iloc[0])
    if statistic == "mean":
        return float(values.mean())
    if statistic == "median":
        return float(values.median())
    if statistic == "max":
        return float(values.max())
    if statistic == "min":
        return float(values.min())
    if statistic == "std":
        return float(values.std(ddof=0)) if len(values) > 1 else 0.0
    if statistic == "count":
        return float(len(values))
    if statistic == "slope":
        if len(values) < 2 or "hours_from_anchor" not in selected:
            return None
        hours = selected["hours_from_anchor"].astype(float)
        valid = hours.notna()
        if valid.sum() < 2 or hours[valid].nunique() < 2:
            return None
        x = torch.tensor(hours[valid].to_numpy(), dtype=torch.float64)
        y = torch.tensor(values[valid].to_numpy(), dtype=torch.float64)
        x_centered = x - x.mean()
        denom = (x_centered * x_centered).sum()
        if denom <= 0:
            return None
        return float((x_centered * (y - y.mean())).sum() / denom)

    raise ValueError(f"Unsupported phenotype statistic: {statistic}")


def extract_phenotype_value(table: pd.DataFrame, spec: PhenotypeQuerySpec) -> Optional[float]:
    if table is None or table.empty:
        return None

    item_text = table["Item"].fillna("").astype(str)
    aliases = [spec.item, *spec.aliases]
    alias_set = {alias.lower() for alias in aliases if alias}
    if spec.item_regex:
        item_mask = item_text.str.contains(spec.item_regex, case=False, regex=True, na=False)
    else:
        item_mask = item_text.str.lower().isin(alias_set)

    category_mask = table["Category"].map(lambda value: category_is_continuous(value, spec.category_regex))
    numeric_values = pd.to_numeric(table["Value"], errors="coerce")
    mask = item_mask & category_mask & numeric_values.notna()
    if spec.unit:
        unit_text = table["Unit"].fillna("").astype(str).str.strip().str.lower()
        mask = mask & (unit_text == spec.unit.strip().lower())
    if not mask.any():
        return None

    anchor_time = table["Time"].dropna().iloc[0] if table["Time"].notna().any() else None
    return aggregate_phenotype_value(table.loc[mask], spec, anchor_time)


class PhenotypeValueExtractor:
    def __init__(self, query_specs: List[PhenotypeQuerySpec]):
        self.query_specs = query_specs
        self.exact_groups: Dict[tuple, List[tuple]] = {}
        self.fallback_indices = []

        for spec_idx, spec in enumerate(query_specs):
            if spec.item_regex or spec.category_regex != "^measurement$":
                self.fallback_indices.append(spec_idx)
                continue
            aliases = tuple(sorted({spec.item.lower(), *(alias.lower() for alias in spec.aliases)}))
            group_key = (aliases, spec.unit.strip().lower(), spec.transform)
            self.exact_groups.setdefault(group_key, []).append((spec_idx, spec))

    def __call__(self, table: pd.DataFrame):
        values = [0.0] * len(self.query_specs)
        masks = [False] * len(self.query_specs)
        if table is None or table.empty:
            return values, masks

        numeric_values = pd.to_numeric(table["Value"], errors="coerce")
        category = table["Category"].fillna("").astype(str).str.strip().str.lower()
        numeric_rows = table.loc[(category == "measurement") & numeric_values.notna()].copy()
        numeric_rows["_item_key"] = numeric_rows["Item"].fillna("").astype(str).str.strip().str.lower()
        numeric_rows["_unit_key"] = numeric_rows["Unit"].fillna("").astype(str).str.strip().str.lower()
        by_item_unit = {
            key: group.drop(columns=["_item_key", "_unit_key"])
            for key, group in numeric_rows.groupby(["_item_key", "_unit_key"], sort=False)
        }
        by_item = {
            key: group.drop(columns=["_item_key", "_unit_key"])
            for key, group in numeric_rows.groupby("_item_key", sort=False)
        }
        anchor_time = table["Time"].dropna().iloc[0] if table["Time"].notna().any() else None

        for (aliases, unit, _), indexed_specs in self.exact_groups.items():
            groups = []
            for alias in aliases:
                selected = by_item_unit.get((alias, unit)) if unit else by_item.get(alias)
                if selected is not None:
                    groups.append(selected)
            if not groups:
                continue
            selected = groups[0] if len(groups) == 1 else pd.concat(groups, ignore_index=True)
            for spec_idx, spec in indexed_specs:
                value = aggregate_phenotype_value(selected, spec, anchor_time)
                if value is not None and math.isfinite(float(value)):
                    values[spec_idx] = float(value)
                    masks[spec_idx] = True

        for spec_idx in self.fallback_indices:
            value = extract_phenotype_value(table, self.query_specs[spec_idx])
            if value is not None and math.isfinite(float(value)):
                values[spec_idx] = float(value)
                masks[spec_idx] = True

        return values, masks


def discover_numeric_query_specs(
    records: List[Dict[str, Any]],
    datasets: List[Dataset],
    min_occurrence: int,
    max_phenotypes: int,
    statistics: List[str],
    time_windows: List[str],
    category_regex: str,
    num_workers: int = 1,
) -> List[PhenotypeQuerySpec]:
    def merge_counts(
        destination: Dict[str, Dict[str, Any]],
        source: Dict[str, Dict[str, Any]],
    ) -> None:
        for key, value in source.items():
            entry = destination.setdefault(
                key,
                {"item": value["item"], "unit": value["unit"], "count": 0},
            )
            entry["count"] += int(value["count"])

    counts: Dict[str, Dict[str, Any]] = {}
    if num_workers <= 1 or len(records) < 2:
        merge_counts(counts, _count_numeric_measurements(records, datasets, category_regex))
    else:
        worker_count = min(int(num_workers), len(records))
        chunk_size = max(1, math.ceil(len(records) / (worker_count * 4)))
        chunks = [records[start : start + chunk_size] for start in range(0, len(records), chunk_size)]
        context = mp.get_context("fork")
        with context.Pool(
            processes=worker_count,
            initializer=_init_discovery_worker,
            initargs=(datasets, category_regex),
        ) as pool:
            for partial_counts in tqdm(
                pool.imap_unordered(_count_numeric_measurement_chunk, chunks),
                total=len(chunks),
                desc="Discovering continuous phenotypes",
                disable=not is_rank0(),
            ):
                merge_counts(counts, partial_counts)

    candidates = [
        value
        for value in counts.values()
        if int(value["count"]) >= min_occurrence
    ]
    candidates.sort(key=lambda value: (-int(value["count"]), str(value["item"]).lower()))
    if max_phenotypes > 0:
        candidates = candidates[:max_phenotypes]

    windows = [parse_time_window(window) for window in time_windows]
    specs = []
    for candidate in candidates:
        for window in windows:
            for statistic in statistics:
                raw_spec = {
                    "item": candidate["item"],
                    "aliases": [candidate["item"]],
                    "unit": candidate["unit"],
                    "statistic": statistic,
                    "window_name": window["name"],
                    "window_start_hours": window["start"],
                    "window_end_hours": window["end"],
                    "category_regex": category_regex,
                }
                specs.append(make_query_spec(raw_spec))
    if not specs:
        raise ValueError("No continuous phenotypes discovered. Provide --phenotype_spec_path or relax discovery filters.")
    return specs


_DISCOVERY_DATASETS: Optional[List[Dataset]] = None
_DISCOVERY_CATEGORY_REGEX = ""


def _init_discovery_worker(datasets: List[Dataset], category_regex: str) -> None:
    global _DISCOVERY_DATASETS, _DISCOVERY_CATEGORY_REGEX
    _DISCOVERY_DATASETS = datasets
    _DISCOVERY_CATEGORY_REGEX = category_regex


def _count_numeric_measurements(
    records: List[Dict[str, Any]],
    datasets: List[Dataset],
    category_regex: str,
) -> Dict[str, Dict[str, Any]]:
    counts: Dict[str, Dict[str, Any]] = {}
    for record in records:
        sample = datasets[record["dataset_idx"]][record["sample_idx"]]
        table = normalize_table(sample.get("measurement_table"), max_table_len=None)
        if table is None or table.empty:
            continue
        category_mask = table["Category"].map(lambda value: category_is_continuous(value, category_regex))
        numeric_values = pd.to_numeric(table["Value"], errors="coerce")
        numeric_rows = table[category_mask & numeric_values.notna()]
        for item, unit in numeric_rows[["Item", "Unit"]].fillna("").astype(str).itertuples(index=False):
            item = " ".join(item.split())
            unit = " ".join(unit.split())
            if not item or item.lower() == "nan":
                continue
            key = f"{item.lower()}|{unit.lower()}"
            entry = counts.setdefault(key, {"item": item, "unit": unit, "count": 0})
            entry["count"] += 1
    return counts


def _count_numeric_measurement_chunk(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if _DISCOVERY_DATASETS is None:
        raise RuntimeError("Discovery worker was not initialized.")
    return _count_numeric_measurements(records, _DISCOVERY_DATASETS, _DISCOVERY_CATEGORY_REGEX)


def local_rank0() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def wait_for_query_cache(cache_path: str, query_keys: Iterable[str]) -> Dict[str, Any]:
    query_keys = list(query_keys)
    while True:
        if os.path.exists(cache_path):
            try:
                cache = torch.load(cache_path, map_location="cpu", weights_only=False)
                cached_embeddings = cache["embeddings"]
                if all(query_key in cached_embeddings for query_key in query_keys):
                    return cache
            except (EOFError, RuntimeError, OSError):
                pass
        time.sleep(2)


def tokenizer_path_for_knowledge_encoder(model_path: str, base_model_path: str) -> str:
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
        return model_path
    if os.path.isfile(model_path):
        parent = os.path.dirname(model_path)
        if os.path.exists(os.path.join(parent, "tokenizer_config.json")):
            return parent
    return base_model_path


def load_knowledge_encoder(model_path: str, base_model_path: str, device: torch.device) -> TextEncoder:
    checkpoint_path = resolve_checkpoint_path(model_path)
    if checkpoint_path is None:
        model = TextEncoder(model_path)
        return model.to(device).eval()

    model = TextEncoder(base_model_path)
    state_dict = strip_module_prefix(load_checkpoint_state_dict(checkpoint_path))
    model_state = model.state_dict()
    matched_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    model.load_state_dict(matched_state, strict=False)
    print(f"Loaded knowledge encoder checkpoint: {checkpoint_path} (matched={len(matched_state)})")
    return model.to(device).eval()


def encode_knowledge_query_texts(
    query_texts: Dict[str, str],
    model_path: str,
    base_model_path: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    query_keys = sorted(query_texts)
    if not query_keys:
        return {}

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path_for_knowledge_encoder(model_path, base_model_path),
        use_fast=True,
    )
    model = load_knowledge_encoder(model_path, base_model_path, device)
    texts = [query_texts[query_key] for query_key in query_keys]
    embeddings = {}
    for start in tqdm(
        range(0, len(query_keys), batch_size),
        desc="Encoding phenotype queries",
        disable=not show_progress,
    ):
        end = min(start + batch_size, len(query_keys))
        tokens = tokenizer(
            texts[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            query_embeds = model.encode_text(tokens).float().cpu()
        for offset, query_key in enumerate(query_keys[start:end]):
            embeddings[query_key] = query_embeds[offset]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


def build_knowledge_query_embeddings(
    query_texts: Dict[str, str],
    cache_path: str,
    model_path: str,
    base_model_path: str,
    max_length: int,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    query_texts = {str(key): str(text) for key, text in query_texts.items()}
    query_keys = sorted(query_texts.keys())

    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_embeddings = cache["embeddings"]
        if all(query_key in cached_embeddings for query_key in query_keys):
            return {query_key: cached_embeddings[query_key].float() for query_key in query_keys}

    if not local_rank0():
        cache = wait_for_query_cache(cache_path, query_keys)
        cached_embeddings = cache["embeddings"]
        return {query_key: cached_embeddings[query_key].float() for query_key in query_keys}

    embeddings = {}
    if os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        embeddings.update({key: value.float() for key, value in cache.get("embeddings", {}).items()})

    missing_keys = [query_key for query_key in query_keys if query_key not in embeddings]
    if missing_keys:
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu")
        missing_query_texts = {query_key: query_texts[query_key] for query_key in missing_keys}
        embeddings.update(
            encode_knowledge_query_texts(
                query_texts=missing_query_texts,
                model_path=model_path,
                base_model_path=base_model_path,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
                show_progress=is_rank0(),
            )
        )

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    text_dim = int(next(iter(embeddings.values())).numel())
    tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": text_dim,
            "model_path": model_path,
            "base_model_path": base_model_path,
        },
        tmp_cache_path,
    )
    os.replace(tmp_cache_path, cache_path)
    return {query_key: embeddings[query_key].float() for query_key in query_keys}


def load_precomputed_query_embeddings(
    query_texts: Dict[str, str],
    cache_path: str,
) -> Dict[str, torch.Tensor]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Query embedding cache not found: {cache_path}. "
            "Run scripts/preprocess/precompute_phenotype_queries.sh first."
        )

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    cached_embeddings = cache.get("embeddings", {})
    missing_keys = sorted(set(query_texts) - set(cached_embeddings))
    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        raise ValueError(
            f"Query embedding cache is missing {len(missing_keys)} queries "
            f"(examples: {preview}). Run scripts/preprocess/precompute_phenotype_queries.sh again."
        )
    return {query_key: cached_embeddings[query_key].float() for query_key in query_texts}


@dataclass
class DataArguments:
    dataset: List[str] = field(default_factory=lambda: ["mimic_iv"])
    root_dir: str = field(default="/data/zikun_workspace/mimic-iv-3.1_tabular")
    sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/next_token_prediction.csv"
    )
    val_sample_info_path: str = field(
        default="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val/next_token_prediction.csv"
    )
    table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt"]
    )
    eicu_root_dir: str = field(default="/data/zikun_workspace/eicu-crd")
    eicu_processed_dir: str = field(default="/data/zikun_workspace/eicu-crd/processed")
    eicu_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json")
    eicu_val_sample_info_path: str = field(default="/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json")
    eicu_table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt"]
    )
    ehrshot_root_dir: str = field(default="/data/EHR_data_public/EHRSHOT")
    ehrshot_sample_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/pretraining_index/sample_info_train.csv")
    ehrshot_val_sample_info_path: str = field(default="/data/EHR_data_public/EHRSHOT/pretraining_index/sample_info_val.csv")
    ehrshot_table_text_embedding: List[str] = field(
        default_factory=lambda: ["/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt"]
    )
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    pretrained_path: Optional[str] = field(default="/data/zikun_workspace/checkpoints/pretraining/task_query_classification")
    phenotype_spec_path: str = field(default="")
    query_embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/phenotype_metric_learning/knowledge_query_embeddings.pt"
    )
    knowledge_encoder_path: str = field(
        default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt"
    )
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=128)
    query_embedding_batch_size: int = field(default=256)
    precomputed_queries_only: bool = field(default=False)
    preprocessed_input_dir: str = field(
        default="/data/zikun_workspace/.cache/phenotype_metric_learning/inputs"
    )
    preprocessed_inputs_only: bool = field(default=False)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_table_len: Optional[int] = field(default=4096)
    min_table_rows: int = field(default=2)
    auto_discover_phenotypes: bool = field(default=True)
    min_phenotype_occurrence: int = field(default=50)
    max_auto_phenotypes: int = field(default=256)
    phenotype_statistics: List[str] = field(default_factory=lambda: ["latest"])
    phenotype_time_windows: List[str] = field(default_factory=lambda: ["full::"])
    phenotype_category_regex: str = field(default="^measurement$")


@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    output_dir: str = field(default="/data/zikun_workspace/checkpoints/pretraining/phenotype_metric_learning")
    num_train_epochs: int = field(default=1)
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
    dataloader_num_workers: int = field(default=16)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="wandb")
    wandb_project: Optional[str] = field(default="Phenotype_Metric_Learning")
    metric_for_best_model: str = field(default="eval_mae")
    greater_is_better: bool = field(default=False)
    huber_delta: float = field(default=1.0)
    projection_loss_weight: float = field(default=1.0)
    transe_loss_weight: float = field(default=0.0)
    relation_l2_weight: float = field(default=0.0)
    min_pair_delta: float = field(default=0.0)
    min_lr_ratio: float = field(default=0.1)

    def __post_init__(self):
        super().__post_init__()
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
        self.eval_strategy = "steps"
        self.load_best_model_at_end = True


class PhenotypeMetricDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        datasets: List[Dataset],
        max_table_len: Optional[int],
        is_eval: bool = False,
    ):
        self.records = records
        self.datasets = datasets
        self.max_table_len = max_table_len
        self.is_eval = is_eval

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        sample = self.datasets[record["dataset_idx"]][record["sample_idx"]]
        table = normalize_table(sample.get("measurement_table"), self.max_table_len)
        if table is None:
            table = pd.DataFrame(columns=["Time", "Item", "Value", "Unit", "Category"])
        return {
            "sample_key": record["sample_key"],
            "subject_id": record["subject_id"],
            "table": table,
            "is_eval": self.is_eval,
        }


class PhenotypeMetricCollator:
    def __init__(
        self,
        text_to_idx: Dict[str, int],
        type_vocab: Dict[str, int],
        query_specs: List[PhenotypeQuerySpec],
        max_table_len: Optional[int],
        min_table_rows: int,
        augmentation_seed: int,
    ):
        self.text_to_idx = text_to_idx
        self.type_vocab = type_vocab
        self.query_specs = query_specs
        self.value_extractor = PhenotypeValueExtractor(query_specs)
        self.max_table_len = max_table_len
        self.min_table_rows = min_table_rows
        self.augmentation_seed = augmentation_seed
        self._batch_counter = 0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        tables = []
        value_rows = []
        mask_rows = []
        batch_counter = self._batch_counter
        self._batch_counter += 1
        worker_info = get_worker_info()
        worker_seed = int(worker_info.seed) if worker_info is not None else 0

        for sample in batch:
            table = sample["table"]
            if table is None or table.empty:
                continue
            if self.max_table_len is not None:
                table = table.tail(self.max_table_len).reset_index(drop=True)
            if len(table) < self.min_table_rows:
                continue

            # Keep the table unchanged for deterministic eval; the seed is reserved
            # for future phenotype-aware table augmentations.
            _ = stable_seed(f"{sample['sample_key']}|{worker_seed}|{batch_counter}", self.augmentation_seed)

            values, masks = self.value_extractor(table)

            tables.append(table)
            value_rows.append(torch.tensor(values, dtype=torch.float))
            mask_rows.append(torch.tensor(masks, dtype=torch.bool))

        if len(tables) == 0:
            raise ValueError("All samples in this batch are invalid after phenotype metric collation.")

        table_tensors = build_table_token_tensors(
            tables,
            text_to_idx=self.text_to_idx,
            pad_idx=0,
            type_vocab=self.type_vocab,
        )
        table_tensors["phenotype_values"] = torch.stack(value_rows, dim=0)
        table_tensors["phenotype_mask"] = torch.stack(mask_rows, dim=0)
        table_tensors["labels"] = torch.zeros(len(tables), dtype=torch.long)
        return table_tensors


class PreprocessedPhenotypeMetricDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        split: str,
        query_specs: List[PhenotypeQuerySpec],
        text_to_idx: Dict[str, int],
    ):
        self.split_dir = os.path.join(cache_root, split)
        manifest_path = os.path.join(self.split_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Preprocessed {split} input manifest not found: {manifest_path}. "
                "Run scripts/preprocess/build_phenotype_metric_samples.sh first."
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        if int(self.manifest.get("format_version", -1)) != PREPROCESSED_INPUT_FORMAT_VERSION:
            raise ValueError(f"Unsupported preprocessed input format in {manifest_path}.")
        expected_spec_fingerprint = phenotype_spec_fingerprint(query_specs)
        if self.manifest.get("phenotype_spec_fingerprint") != expected_spec_fingerprint:
            raise ValueError(
                f"Phenotype query specs do not match the preprocessed {split} inputs. "
                "Re-run scripts/preprocess/build_phenotype_metric_samples.sh."
            )
        expected_vocab_fingerprint = text_vocab_fingerprint(text_to_idx)
        if self.manifest.get("text_vocab_fingerprint") != expected_vocab_fingerprint:
            raise ValueError(
                f"Table text vocabulary does not match the preprocessed {split} inputs. "
                "Re-run scripts/preprocess/build_phenotype_metric_samples.sh."
            )

        self.num_queries = len(query_specs)
        self.parts = list(self.manifest.get("parts", []))
        self.part_ends = []
        total = 0
        for part in self.parts:
            total += int(part["sample_count"])
            self.part_ends.append(total)
        self.sample_count = total
        self._open_parts = {}
        if self.sample_count == 0:
            raise ValueError(f"Preprocessed {split} input cache contains no valid samples.")
        print(
            f"Loaded preprocessed {split} inputs: "
            f"{self.sample_count} episodes across {len(self.parts)} parts"
        )

    def __len__(self) -> int:
        return self.sample_count

    def _open_part(self, part_idx: int):
        if part_idx in self._open_parts:
            return self._open_parts[part_idx]

        part = self.parts[part_idx]
        part_dir = os.path.join(self.split_dir, part["path"])
        sample_count = int(part["sample_count"])
        total_rows = int(part["total_rows"])
        arrays = {
            field_name: np.memmap(
                os.path.join(part_dir, f"{field_name}.bin"),
                dtype=np.dtype(dtype),
                mode="r",
                shape=(total_rows,),
            )
            for field_name, dtype in PREPROCESSED_SEQUENCE_DTYPES.items()
        }
        opened = {
            "offsets": np.load(os.path.join(part_dir, "offsets.npy"), mmap_mode="r"),
            "arrays": arrays,
            "phenotype_values": np.memmap(
                os.path.join(part_dir, "phenotype_values.bin"),
                dtype=np.float32,
                mode="r",
                shape=(sample_count, self.num_queries),
            ),
            "phenotype_mask": np.memmap(
                os.path.join(part_dir, "phenotype_mask.bin"),
                dtype=np.uint8,
                mode="r",
                shape=(sample_count, self.num_queries),
            ),
        }
        self._open_parts[part_idx] = opened
        return opened

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0:
            idx += self.sample_count
        if idx < 0 or idx >= self.sample_count:
            raise IndexError(idx)

        part_idx = bisect.bisect_right(self.part_ends, idx)
        part_start = 0 if part_idx == 0 else self.part_ends[part_idx - 1]
        local_idx = idx - part_start
        opened = self._open_part(part_idx)
        row_start = int(opened["offsets"][local_idx])
        row_end = int(opened["offsets"][local_idx + 1])

        sample = {
            field_name: torch.from_numpy(
                np.asarray(array[row_start:row_end]).copy()
            )
            for field_name, array in opened["arrays"].items()
        }
        sample["phenotype_values"] = torch.from_numpy(
            np.asarray(opened["phenotype_values"][local_idx]).copy()
        )
        sample["phenotype_mask"] = torch.from_numpy(
            np.asarray(opened["phenotype_mask"][local_idx]).copy()
        ).bool()
        return sample


class PreprocessedPhenotypeMetricCollator:
    def __init__(self, max_table_len: Optional[int], min_table_rows: int):
        self.max_table_len = max_table_len
        self.min_table_rows = min_table_rows

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        kept_samples = []
        for sample in batch:
            sequence_length = int(sample["item_ids"].numel())
            if self.max_table_len is not None:
                sequence_length = min(sequence_length, int(self.max_table_len))
            if sequence_length >= self.min_table_rows:
                kept_samples.append((sample, sequence_length))
        if not kept_samples:
            raise ValueError("All preprocessed samples in this batch are too short after truncation.")

        batch_size = len(kept_samples)
        padded_length = max(sequence_length for _, sequence_length in kept_samples)
        table_tensors = {
            "item_ids": torch.zeros(batch_size, padded_length, dtype=torch.long),
            "unit_ids": torch.zeros(batch_size, padded_length, dtype=torch.long),
            "value_text_ids": torch.zeros(batch_size, padded_length, dtype=torch.long),
            "times": torch.zeros(batch_size, padded_length, dtype=torch.float),
            "numeric_values": torch.zeros(batch_size, padded_length, dtype=torch.float),
            "numeric_mask": torch.zeros(batch_size, padded_length, dtype=torch.float),
            "seq_mask": torch.zeros(batch_size, padded_length, dtype=torch.float),
            "type_ids": torch.zeros(batch_size, padded_length, dtype=torch.long),
        }

        for row_idx, (sample, sequence_length) in enumerate(kept_samples):
            source_start = int(sample["item_ids"].numel()) - sequence_length
            for field_name in ("item_ids", "unit_ids", "value_text_ids", "type_ids"):
                table_tensors[field_name][row_idx, :sequence_length] = sample[field_name][
                    source_start:
                ].long()
            for field_name in ("numeric_values", "numeric_mask"):
                table_tensors[field_name][row_idx, :sequence_length] = sample[field_name][
                    source_start:
                ].float()

            times = sample["times"][source_start:].float().clone()
            valid_times = times > 0
            if valid_times.any():
                times[valid_times] = times[valid_times] - times[valid_times][0] + 1.0
            table_tensors["times"][row_idx, :sequence_length] = times
            table_tensors["seq_mask"][row_idx, :sequence_length] = 1.0

        table_tensors["phenotype_values"] = torch.stack(
            [sample["phenotype_values"].float() for sample, _ in kept_samples]
        )
        table_tensors["phenotype_mask"] = torch.stack(
            [sample["phenotype_mask"].bool() for sample, _ in kept_samples]
        )
        table_tensors["labels"] = torch.zeros(batch_size, dtype=torch.long)
        return table_tensors


class PhenotypeMetricTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer] = None):
        if self.lr_scheduler is None and str(self.args.lr_scheduler_type) == "cosine":
            optimizer = self.optimizer if optimizer is None else optimizer
            warmup_steps = self.args.get_warmup_steps(num_training_steps)
            min_lr_ratio = float(getattr(self.args, "min_lr_ratio", 0.0))

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return self.lr_scheduler

        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("labels", None)
        loss, outputs = model(**inputs)
        if return_outputs:
            return loss, outputs
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataloader = self.get_eval_dataloader(eval_dataset if eval_dataset is not None else self.eval_dataset)
        self.model.eval()

        totals = torch.zeros(4, dtype=torch.float64, device=self.args.device)
        for inputs in tqdm(
            dataloader,
            desc=f"Eval step {self.state.global_step}",
            disable=not is_rank0(),
            dynamic_ncols=True,
            leave=False,
        ):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                _, outputs = self.compute_loss(self.model, inputs, return_outputs=True)
            totals[0] += outputs["loss_sum"].double()
            totals[1] += outputs["abs_error_sum"].double()
            totals[2] += outputs["squared_error_sum"].double()
            totals[3] += outputs["pair_count"].double()

        if is_distributed():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)

        pair_count = max(float(totals[3].item()), 1.0)
        metrics = {
            f"{metric_key_prefix}_loss": float(totals[0].item() / pair_count),
            f"{metric_key_prefix}_mae": float(totals[1].item() / pair_count),
            f"{metric_key_prefix}_rmse": float(math.sqrt(totals[2].item() / pair_count)),
            f"{metric_key_prefix}_pair_count": float(totals[3].item()),
        }
        if is_rank0():
            print(f"[Eval] step={self.state.global_step} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self.model.train()
        return metrics


def build_state_key(dataset_name: str, sample_info: Dict[str, Any]) -> str:
    if sample_info.get("sample_id") is not None:
        return str(sample_info["sample_id"])
    if dataset_name == "mimic_iv":
        return (
            f"mimic_iv|{sample_info.get('subject_id', '')}|"
            f"{sample_info.get('hadm_id', sample_info.get('stay_id', ''))}|"
            f"{sample_info.get('context_begin', '')}|"
            f"{sample_info.get('context_end', '')}"
        )
    if dataset_name == "eicu":
        return (
            f"eicu|{sample_info.get('patient_id', '')}|"
            f"{sample_info.get('icustay_id', sample_info.get('patientunitstayid', ''))}|"
            f"{sample_info.get('task_name', '')}|"
            f"{sample_info.get('obs_hours', '')}"
        )
    if dataset_name == "ehrshot":
        return (
            f"ehrshot|{sample_info.get('patient_id', '')}|"
            f"{sample_info.get('task_name', '')}|"
            f"{sample_info.get('prediction_time', '')}"
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_subject_id(dataset_name: str, sample_info: Dict[str, Any]) -> str:
    if dataset_name == "mimic_iv":
        return str(sample_info.get("subject_id", ""))
    if dataset_name in {"eicu", "ehrshot"}:
        return str(sample_info.get("patient_id", ""))
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_one_dataset(dataset_name: str, data_args: DataArguments, sample_info_path: str):
    if dataset_name == "mimic_iv":
        return MIMICIV(
            root_dir=data_args.root_dir,
            sample_info_path=sample_info_path,
            lazy_mode=True,
            shuffle=False,
            max_samples=None,
            use_table_length_cache=False,
        )
    if dataset_name == "eicu":
        return EICUDataset(
            root_dir=data_args.eicu_root_dir,
            processed_dir=data_args.eicu_processed_dir,
            sample_info_path=sample_info_path,
            task_name=None,
            lazy_mode=True,
            shuffle=False,
        )
    if dataset_name == "ehrshot":
        return EHRSHOTDataset(
            root_dir=data_args.ehrshot_root_dir,
            sample_info_path=sample_info_path,
            task_name=None,
            lazy_mode=True,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def split_sample_info_path(data_args: DataArguments, dataset_name: str, split: str) -> str:
    if dataset_name == "mimic_iv":
        return data_args.sample_info_path if split == "train" else data_args.val_sample_info_path
    if dataset_name == "eicu":
        return data_args.eicu_sample_info_path if split == "train" else data_args.eicu_val_sample_info_path
    if dataset_name == "ehrshot":
        return data_args.ehrshot_sample_info_path if split == "train" else data_args.ehrshot_val_sample_info_path
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_split_records(data_args: DataArguments, split: str):
    records = []
    datasets = []
    max_samples = data_args.max_train_samples if split == "train" else data_args.max_eval_samples
    for dataset_name in data_args.dataset:
        dataset = build_one_dataset(dataset_name, data_args, split_sample_info_path(data_args, dataset_name, split))
        dataset_idx = len(datasets)
        datasets.append(dataset)
        dataset_records = []
        for sample_idx, sample_info in enumerate(dataset.sample_info):
            dataset_records.append(
                {
                    "dataset_idx": dataset_idx,
                    "sample_idx": sample_idx,
                    "sample_key": build_state_key(dataset_name, sample_info),
                    "subject_id": build_subject_id(dataset_name, sample_info),
                }
            )
        if max_samples is not None:
            dataset_records = dataset_records[:max_samples]
        print(f"{split} {dataset_name} episode samples: {len(dataset_records)}")
        records.extend(dataset_records)
    return records, datasets


def get_embedding_cache_paths(data_args: DataArguments) -> List[str]:
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


def build_query_specs(data_args: DataArguments, train_records, train_datasets) -> List[PhenotypeQuerySpec]:
    if data_args.phenotype_spec_path:
        specs = load_query_specs(data_args.phenotype_spec_path)
        print(f"Loaded phenotype query specs: {len(specs)}")
        return specs
    if not data_args.auto_discover_phenotypes:
        raise ValueError("Set --phenotype_spec_path or enable --auto_discover_phenotypes.")
    specs = discover_numeric_query_specs(
        records=train_records,
        datasets=train_datasets,
        min_occurrence=data_args.min_phenotype_occurrence,
        max_phenotypes=data_args.max_auto_phenotypes,
        statistics=data_args.phenotype_statistics,
        time_windows=data_args.phenotype_time_windows,
        category_regex=data_args.phenotype_category_regex,
    )
    print(f"Auto-discovered phenotype-time query specs: {len(specs)}")
    return specs


def main():
    parser = HfArgumentParser((DataArguments, TrainingArgumentsCustom))
    data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ.setdefault("MIMIC_SKIP_SAMPLE_CACHE_CHECK", "1")
    set_seed(training_args.seed)

    print("Query-conditioned continuous phenotype metric learning")
    print(f"Datasets: {', '.join(data_args.dataset)}")
    print(f"Pretrained path: {data_args.pretrained_path}")
    print(f"Knowledge encoder: {data_args.knowledge_encoder_path}")

    if data_args.preprocessed_inputs_only:
        if not data_args.phenotype_spec_path:
            raise ValueError("--phenotype_spec_path is required with --preprocessed_inputs_only true.")
        train_records = train_datasets = val_records = val_datasets = None
        query_specs = load_query_specs(data_args.phenotype_spec_path)
        print(f"Loaded phenotype query specs: {len(query_specs)}")
    else:
        train_records, train_datasets = build_split_records(data_args, "train")
        val_records, val_datasets = build_split_records(data_args, "val")
        query_specs = build_query_specs(data_args, train_records, train_datasets)

    query_texts = {spec.key: spec.query_text for spec in query_specs}
    if data_args.precomputed_queries_only:
        query_embeddings = load_precomputed_query_embeddings(
            query_texts=query_texts,
            cache_path=data_args.query_embedding_cache,
        )
    else:
        query_embeddings = build_knowledge_query_embeddings(
            query_texts=query_texts,
            cache_path=data_args.query_embedding_cache,
            model_path=data_args.knowledge_encoder_path,
            base_model_path=data_args.knowledge_encoder_base_model_path,
            max_length=data_args.query_max_length,
            batch_size=data_args.query_embedding_batch_size,
        )
    query_embedding_matrix = torch.stack([query_embeddings[spec.key] for spec in query_specs], dim=0)
    phenotype_scales = torch.tensor(
        [float(spec.scale) if spec.scale is not None and float(spec.scale) > 0 else 0.0 for spec in query_specs],
        dtype=torch.float,
    )
    print(f"Phenotype-time queries: {len(query_specs)}")
    print(f"Knowledge query dim: {query_embedding_matrix.size(-1)}")

    text_dim, text_to_idx, embedding_matrix = load_table_embeddings(get_embedding_cache_paths(data_args))
    type_vocab = load_type_vocab(data_args.type_vocab_file)
    print(f"Table text dim: {text_dim}")

    config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=max(type_vocab.values()) + 1,
        max_table_len=data_args.max_table_len,
        dim_out=int(query_embedding_matrix.size(-1)),
    )
    model = PhenotypeMetricModel(
        config=config,
        embedding_matrix=embedding_matrix,
        query_embedding_matrix=query_embedding_matrix,
        phenotype_scales=phenotype_scales,
        huber_delta=training_args.huber_delta,
        projection_loss_weight=training_args.projection_loss_weight,
        transe_loss_weight=training_args.transe_loss_weight,
        relation_l2_weight=training_args.relation_l2_weight,
        min_pair_delta=training_args.min_pair_delta,
    )
    model = load_matching_weights(model, data_args.pretrained_path)

    if data_args.preprocessed_inputs_only:
        train_dataset = PreprocessedPhenotypeMetricDataset(
            cache_root=data_args.preprocessed_input_dir,
            split="train",
            query_specs=query_specs,
            text_to_idx=text_to_idx,
        )
        val_dataset = PreprocessedPhenotypeMetricDataset(
            cache_root=data_args.preprocessed_input_dir,
            split="val",
            query_specs=query_specs,
            text_to_idx=text_to_idx,
        )
        collator = PreprocessedPhenotypeMetricCollator(
            max_table_len=data_args.max_table_len,
            min_table_rows=data_args.min_table_rows,
        )
    else:
        train_dataset = PhenotypeMetricDataset(
            records=train_records,
            datasets=train_datasets,
            max_table_len=data_args.max_table_len,
            is_eval=False,
        )
        val_dataset = PhenotypeMetricDataset(
            records=val_records,
            datasets=val_datasets,
            max_table_len=data_args.max_table_len,
            is_eval=True,
        )
        collator = PhenotypeMetricCollator(
            text_to_idx=text_to_idx,
            type_vocab=type_vocab,
            query_specs=query_specs,
            max_table_len=data_args.max_table_len,
            min_table_rows=data_args.min_table_rows,
            augmentation_seed=training_args.seed,
        )

    trainer = PhenotypeMetricTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
