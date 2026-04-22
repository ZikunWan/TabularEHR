import os
import sys
import json
import builtins
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Set
from collections import defaultdict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from safetensors.torch import save_file

from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    EarlyStoppingCallback,
    PreTrainedModel,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.TableEncoder.encoder import LongTableEncoder
from models.TableEncoder.config import TableEncoderConfig
from dataset.mimic.mimic_dataset import MIMICIV
from utils.collate import build_table_token_tensors


def is_rank0() -> bool:
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            pass
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def rank0_print(*args, **kwargs):
    if is_rank0():
        builtins.print(*args, **kwargs)


print = rank0_print


# ======================== Cross-GPU Gather ========================
class GatherWithGrad(torch.autograd.Function):
    """All-gather that allows gradients to flow back to the local rank.

    Forward:  gather padded tensors from all ranks
    Backward: each rank receives only its own slice of the upstream gradient
    """

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        output = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor.contiguous())
        ctx.rank = dist.get_rank()
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        return grads[ctx.rank]


def all_gather_with_grad(tensor: torch.Tensor) -> tuple:
    """Gather embeddings from all GPUs into a single tensor, handling uneven batch sizes.
    
    Returns:
        (gathered_tensor, local_label_offset)
    """
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return tensor, 0

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 1. Gather the sizes of all batches to handle uneven batches (e.g., last batch)
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    sizes = [s.item() for s in size_list]
    max_size = max(sizes)
    
    # Calculate offset for labels
    label_offset = sum(sizes[:rank])

    # 2. Pad local tensor to max_size if necessary
    if local_size < max_size:
        padding = torch.zeros((max_size - local_size.item(), *tensor.shape[1:]),
                              dtype=tensor.dtype, device=tensor.device)
        tensor_padded = torch.cat((tensor, padding), dim=0)
    else:
        tensor_padded = tensor

    # 3. Gather
    gathered_padded = GatherWithGrad.apply(tensor_padded)
    
    # 4. Remove padding from gathered tensors and concatenate
    gathered_unpadded = []
    for i, s in enumerate(sizes):
        gathered_unpadded.append(gathered_padded[i][:s])
        
    return torch.cat(gathered_unpadded, dim=0), label_offset


def load_embedding_cache(cache_path: str):
    """Load pre-computed embedding cache."""
    import time
    print(f"Loading embedding cache from {cache_path}...")
    t0 = time.time()
    data = torch.load(cache_path, map_location='cpu')
    embeddings = data['embeddings']
    text_dim = data['text_dim']
    print(f"  ✓ Loaded {len(embeddings)} embeddings (dim={text_dim}) in {time.time()-t0:.2f}s")
    return embeddings, text_dim

@dataclass
class ModelArguments:
    projector_hidden_size: int = field(
        default=2048,
        metadata={"help": "dim_out passed to LongTableEncoder (shared embedding space size)"}
    )


@dataclass
class DataArguments:
    root_dir: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular")
    train_info_path: Optional[str] = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/train/contrastive_learning.csv",
        metadata={"help": "Optional path to train sample_info CSV (preferred)."}
    )
    val_info_path: Optional[str] = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/val/contrastive_learning.csv",
        metadata={"help": "Optional path to val sample_info CSV. If provided, eval uses this file directly."}
    )
    free_text_embedding: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/embeddings/table_free_text/embeddings.pt",
        metadata={
            "help": (
                "Path to the pre-computed free-text embedding file (.pt). "
                "Generated by preprocess/mimic_iv/7_generate_text_embeddings.py using "
                "MIMICIV(return_table=False) sample['input'] — i.e., the table's own free-text representation."
            )
        }
    )
    kept_blocks_path: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/embeddings/table_free_text/kept_blocks.pt",
        metadata={"help": "Path to pre-computed kept block ids (.pt), keyed by sample_key."}
    )
    table_text_embedding: str = field(
        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/embeddings/table_text_embeddings.pt",
        metadata={"help": "Path to pre-computed table text embedding cache (.pt) for vocab lookup"}
    )
    type_vocab_file: Optional[str] = field(
        default="/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/data/type_vocab.json",
        metadata={"help": "Path to unified type vocabulary JSON file."}
    )
    max_samples_per_split: Optional[int] = field(default=None)
    max_table_length: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum table length to keep a sample (requires table_length column in CSV)"}
    )
    sort_by_table_length: bool = field(
        default=False,
        metadata={"help": "Sort the dataset by table_length (ascending) (requires table_length column)"}
    )
    short_table_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Ratio of samples to keep after sorting by table length, e.g. 0.5 for shortest 50%"}
    )
    
    temperature: float = field(default=0.07)
    use_hard_negatives: bool = field(default=True, metadata={"help": "Enable CCS-based hard negative mining"})
    num_negatives: int = field(default=10, metadata={"help": "Total negatives per sample"})
    same_ccs_ratio: float = field(default=0.5, metadata={"help": "50% from same CCS group"})
    similar_ccs_ratio: float = field(default=0.3, metadata={"help": "30% from other CCS groups"})


@dataclass
class TrainingArgumentsCustom(TrainingArguments):
    """Extended training arguments"""
    output_dir: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/contrastive_learning")
    num_train_epochs: int = field(default=100)
    per_device_train_batch_size: int = field(default=512)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=1e-4)
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    logging_steps: int = field(default=10)
    save_total_limit: int = field(default=2)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=False)
    dataloader_num_workers: int = field(default=32)
    remove_unused_columns: bool = field(default=False)
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    run_name: Optional[str] = field(default=None)
    run_project: Optional[str] = field(default=None, metadata={"help": "Wandb project name"})
    # Early stopping
    early_stopping_patience: int = field(
        default=0,
        metadata={"help": "Number of eval steps with no improvement before stopping. 0 = disabled."}
    )
    metric_for_best_model: str = field(
        default="eval_loss",
        metadata={"help": "Metric to monitor for early stopping / best model. E.g. 'eval_loss' or 'eval_mrr'."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.run_project:
            os.environ["WANDB_PROJECT"] = self.run_project
        # load_best_model_at_end is required for EarlyStoppingCallback
        if self.early_stopping_patience > 0:
            self.load_best_model_at_end = True
            self.greater_is_better = not self.metric_for_best_model.endswith("loss")


# ======================== Model ========================
class AttentionPooling(nn.Module):
    """Attention pooling over query tokens."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        scores = self.attention(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class ContrastiveModel(PreTrainedModel):
    """Model for Table ↔ Free-Text Contrastive Learning.

    Positive pair:
        - table_emb : structured EHR table encoded by LongTableEncoder  (anchor)
        - text_emb  : pre-computed embedding of the SAME visit's free text,
                      produced by 7_generate_text_embeddings.py from
                      MIMICIV(return_table=False) sample['input']
    """

    config_class = TableEncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config: TableEncoderConfig, encoder: nn.Module,
                 temperature=0.07, embedding_matrix=None):
        super().__init__(config)
        self.encoder = encoder
        self.temperature = temperature
        pool_hidden_size = config.dim_out if getattr(config, "dim_out", None) else config.dim
        self.table_pooling = AttentionPooling(pool_hidden_size)

        if embedding_matrix is not None:
            self.text_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        else:
            raise ValueError("embedding_matrix must be provided!")

    def encode_text(self, inputs_embeds):
        """Text side: pre-computed embedding is used directly (no further encoding)."""
        return inputs_embeds

    def encode_table(self, item_ids, unit_ids, value_text_ids, times,
                     numeric_values, numeric_mask, seq_mask=None, type_ids=None):
        """Encode tables and pool Q-Former queries to (batch_size, hidden_size)."""
        item_emb = self.text_embedding(item_ids)
        unit_emb = self.text_embedding(unit_ids)
        value_emb = self.text_embedding(value_text_ids)

        query_embeddings = self.encoder(
            item_emb, unit_emb, value_emb,
            times, numeric_values, numeric_mask,
            seq_mask, type_ids=type_ids
        )
        return self.table_pooling(query_embeddings)

    def forward(self, item_ids, unit_ids, value_text_ids, times,
                numeric_values, numeric_mask,
                seq_mask=None, type_ids=None,
                inputs_embeds=None, negative_inputs_embeds=None, **kwargs):
        # Encode and normalize
        table_emb = F.normalize(self.encode_table(
            item_ids, unit_ids, value_text_ids,
            times, numeric_values, numeric_mask, seq_mask, type_ids
        ), dim=-1)
        text_emb = F.normalize(self.encode_text(inputs_embeds), dim=-1)

        batch_size = table_emb.size(0)
        has_neg = (negative_inputs_embeds is not None)

        # ---- Global batch: gather text embeddings from all GPUs ----
        all_text_emb, label_offset = all_gather_with_grad(text_emb)   # (B_global, D)

        if has_neg:
            num_neg = negative_inputs_embeds.size(1)
            neg_embeds_flat = negative_inputs_embeds.reshape(batch_size * num_neg, -1)
            neg_emb = F.normalize(self.encode_text(neg_embeds_flat), dim=-1)
            neg_emb = neg_emb.view(batch_size, num_neg, -1)

            # In-batch similarities against GLOBAL text embeddings
            sim_t2t_in_batch = torch.matmul(table_emb, all_text_emb.t()) / self.temperature
            # Hard negative similarities
            sim_t2t_hard = torch.bmm(table_emb.unsqueeze(1),
                                     neg_emb.transpose(1, 2)).squeeze(1) / self.temperature
            sim_t2t_all = torch.cat([sim_t2t_in_batch, sim_t2t_hard], dim=1)

            labels = label_offset + torch.arange(batch_size, device=sim_t2t_all.device)
            loss = F.cross_entropy(sim_t2t_all, labels)

            with torch.no_grad():
                acc = (sim_t2t_all.argmax(dim=1) == labels).float().mean()
        else:
            sim = torch.matmul(table_emb, all_text_emb.t()) / self.temperature
            labels = label_offset + torch.arange(batch_size, device=sim.device)
            loss = F.cross_entropy(sim, labels)

            with torch.no_grad():
                acc = (sim.argmax(dim=1) == labels).float().mean()

        if not self.training:
            return loss, {'table_embs': table_emb, 'text_embs': text_emb}

        return loss, {'loss': loss.item(), 'accuracy': acc.item()}


# ======================== Dataset ========================

class ContrastiveDataset(Dataset):
    """Dataset for Table ↔ Free-Text contrastive learning.

    - base_dataset   : MIMICIV(return_table=True) — provides structured table data
    - embeddings_path: path to {idx: tensor} dict saved by 7_generate_text_embeddings.py,
                       where the text is sample['input'] from MIMICIV(return_table=False)
    """

    def __init__(self, base_dataset, embeddings_path=None, kept_blocks_path=None, is_eval=False):
        self.base_dataset = base_dataset
        self.embeddings_path = embeddings_path
        self.kept_blocks_path = kept_blocks_path
        self.is_eval = is_eval

        if not self.embeddings_path or not os.path.exists(self.embeddings_path):
            raise ValueError(
                f"Free-text embedding file not found: {self.embeddings_path}\n"
                "Run preprocess/mimic_iv/7_generate_text_embeddings.py first."
            )
        if not self.kept_blocks_path or not os.path.exists(self.kept_blocks_path):
            raise ValueError(
                f"Kept-block file not found: {self.kept_blocks_path}\n"
                "Run preprocess/mimic_iv/7_generate_text_embeddings.py first."
            )

        print(f"📦 Loading pre-computed free-text embeddings from {self.embeddings_path}...")
        import time
        t0 = time.time()
        self.embeddings = torch.load(self.embeddings_path, weights_only=True)
        print(f"   ✓ Loaded {len(self.embeddings)} embeddings in {time.time()-t0:.2f}s")
        print(f"📦 Loading kept-block metadata from {self.kept_blocks_path}...")
        t1 = time.time()
        self.kept_blocks = torch.load(self.kept_blocks_path, weights_only=True)
        print(f"   ✓ Loaded {len(self.kept_blocks)} kept-block records in {time.time()-t1:.2f}s")

    @staticmethod
    def _build_sample_key(sample_info):
        return (
            f"{sample_info.get('subject_id', '')}|"
            f"{sample_info.get('task', '')}|"
            f"{sample_info.get('context_begin', '')}|"
            f"{sample_info.get('context_end', '')}"
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        sample_info = self.base_dataset.sample_info[idx]
        sample_key = self._build_sample_key(sample_info)

        if sample_key in self.embeddings:
            data = self.embeddings[sample_key]
        
            if data.dim() == 2 and data.size(0) == 1:
                data = data.squeeze(0)
            ccs = sample_info.get("primary_diagnosis_ccs", "")
            if pd.isna(ccs):
                ccs = ""
            else:
                ccs = str(ccs)

            return {
                'tables': sample.get('measurement_table', {}),
                'measurement_table_row_block_ids': sample.get('measurement_table_row_block_ids', []),
                'kept_block_ids': self.kept_blocks.get(sample_key, []),
                'inputs_embeds': data,
                'idx': idx,
                'ccs': ccs,
                'is_eval': self.is_eval,
                'sample_key': sample_key,
            }

        raise KeyError(f"Free-text embedding not found for sample_key={sample_key} (idx={idx})")

    def get_text_embedding(self, idx) -> torch.Tensor:
        """Fast path: retrieve the pre-computed free-text embedding for negative sampling."""
        sample_info = self.base_dataset.sample_info[idx]
        sample_key = self._build_sample_key(sample_info)

        if sample_key in self.embeddings:
            data = self.embeddings[sample_key]
            if data.dim() == 2 and data.size(0) == 1:
                data = data.squeeze(0)
            return data

        return torch.zeros(self.embeddings[list(self.embeddings.keys())[0]].shape[-1])


def build_ccs_index(dataset):
    """Build index: CCS code → list of sample indices"""
    ccs_index = defaultdict(list)
    print("🔍 Building CCS index for hard negative mining...")

    sample_info = dataset.base_dataset.sample_info

    for idx in tqdm(range(len(dataset)), desc="Indexing"):
        sample = sample_info[idx]
        ccs = sample.get('primary_diagnosis_ccs')
        if ccs and str(ccs).lower() != 'nan':
            ccs_index[ccs].append(idx)

    print(f"   ✓ Indexed {len(ccs_index)} CCS groups")
    return ccs_index


# ======================== Collate ========================

def create_collate_fn(dataset, ccs_index, args, vocab_keys, type_vocab):
    text_keys = vocab_keys

    _text_to_idx = {t: i for i, t in enumerate(text_keys)}
    _pad_idx = _text_to_idx.get('[PAD]', 0)

    def collate_fn(batch):
        is_eval = batch[0].get('is_eval', False)
        tables_list = [item['tables'] for item in batch]
        row_block_ids_list = [item.get('measurement_table_row_block_ids', []) for item in batch]
        kept_block_ids_list = [item.get('kept_block_ids', []) for item in batch]

        table_tensors = build_table_token_tensors(
            tables_list=tables_list,
            text_to_idx=_text_to_idx,
            pad_idx=_pad_idx,
            row_block_ids_list=row_block_ids_list,
            kept_block_ids_list=kept_block_ids_list,
            type_vocab=type_vocab,
        )
        bs = len(batch)

        # --- Positive free-text embeddings (pre-computed) ---
        inputs_embeds_list = [item['inputs_embeds'] for item in batch]
        if inputs_embeds_list[0].dim() == 1:
            inputs_embeds = torch.stack(inputs_embeds_list)
        else:
            max_tl = max(t.size(0) for t in inputs_embeds_list)
            hidden_dim = inputs_embeds_list[0].size(1)
            inputs_embeds = torch.zeros(bs, max_tl, hidden_dim, dtype=inputs_embeds_list[0].dtype)
            for i, t in enumerate(inputs_embeds_list):
                inputs_embeds[i, :t.size(0), :] = t

        model_inputs = {
            **table_tensors,
            'inputs_embeds': inputs_embeds,
            'labels': torch.zeros(bs, dtype=torch.long)  # DUMMY label to trigger evaluate
        }

        if not args.use_hard_negatives or is_eval:
            return model_inputs

        # --- Hard negative sampling (free-text embeddings) ---
        neg_data_list = []
        n_same = int(args.num_negatives * args.same_ccs_ratio)
        n_similar = int(args.num_negatives * args.similar_ccs_ratio)

        for item in batch:
            idx, ccs = item['idx'], item['ccs']
            negatives_indices = []

            # 1. Same CCS group
            if ccs and ccs in ccs_index:
                candidates = [i for i in ccs_index[ccs] if i != idx]
                if candidates:
                    negatives_indices.extend(random.sample(candidates, min(n_same, len(candidates))))

            # 2. Similar CCS groups
            other_groups = [c for c in ccs_index if c != ccs]
            if other_groups and n_similar > 0:
                sampled_groups = random.sample(other_groups, min(3, len(other_groups)))
                candidates = [i for g in sampled_groups for i in ccs_index[g]]
                if candidates:
                    negatives_indices.extend(random.sample(candidates, min(n_similar, len(candidates))))

            # 3. Random fill
            attempts = 0
            while len(negatives_indices) < args.num_negatives and attempts < 100:
                cand = random.randint(0, len(dataset) - 1)
                if cand != idx and cand not in negatives_indices:
                    negatives_indices.append(cand)
                attempts += 1

            while len(negatives_indices) < args.num_negatives:
                negatives_indices.append((idx + 1) % len(dataset))

            final_indices = negatives_indices[:args.num_negatives]
            neg_embeds_for_sample = [dataset.get_text_embedding(i) for i in final_indices]
            neg_data_list.append(neg_embeds_for_sample)

        # Stack negatives
        all_neg_tensors = [opt for batch_neg in neg_data_list for opt in batch_neg]

        if all_neg_tensors[0].dim() == 1:
            neg_embeds = torch.stack(all_neg_tensors).view(bs, args.num_negatives, -1)
        else:
            max_nl = max(t.size(0) for t in all_neg_tensors)
            hidden_dim = all_neg_tensors[0].size(1)
            neg_embeds = torch.zeros(bs, args.num_negatives, max_nl, hidden_dim,
                                     dtype=all_neg_tensors[0].dtype)
            for b_i, batch_negs in enumerate(neg_data_list):
                for n_i, t in enumerate(batch_negs):
                    neg_embeds[b_i, n_i, :t.size(0), :] = t

        model_inputs['negative_inputs_embeds'] = neg_embeds
        return model_inputs

    return collate_fn


# ======================== Trainer ========================

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = model(**inputs)
        if return_outputs:
            # outputs contains table_embs/text_embs during eval,
            # or loss/accuracy scalars during train — handle both
            if isinstance(outputs, dict) and 'table_embs' in outputs:
                return loss, (outputs['table_embs'], outputs['text_embs'])
            return (loss, outputs)
        return loss
    
    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        # 调用父类真实的 training_step 走完前向/反向传播
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # 强制把要交给 tr_loss 累加的张量除以累加步数
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        return loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Switch to eval mode
        model = self.model
        model.eval()

        all_table_embs = []
        all_text_embs = []
        total_loss = 0.0
        num_batches = 0

        for step, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, (table_embs, text_embs) = self.compute_loss(
                    model, inputs, return_outputs=True
                )
            total_loss += loss.mean().item()
            num_batches += 1
            all_table_embs.append(table_embs.detach().float().cpu())
            all_text_embs.append(text_embs.detach().float().cpu())

        # Back to train mode
        model.train()

        # Concat this rank's embeddings
        local_table = torch.cat(all_table_embs, dim=0)   # (N_local, D)
        local_text  = torch.cat(all_text_embs,  dim=0)   # (N_local, D)

        # Gather across all ranks
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank       = torch.distributed.get_rank()

            # all_gather_object handles variable sizes (uneven last batch)
            table_list = [None] * world_size
            text_list  = [None] * world_size
            torch.distributed.all_gather_object(table_list, local_table)
            torch.distributed.all_gather_object(text_list,  local_text)

            if rank == 0:
                table_all = torch.cat(table_list, dim=0)
                text_all  = torch.cat(text_list,  dim=0)
            else:
                table_all = local_table   # unused on non-zero ranks
                text_all  = local_text
        else:
            rank      = 0
            table_all = local_table
            text_all  = local_text

        # Average loss across ranks (all_reduce requires CUDA tensor with NCCL)
        loss_tensor = torch.tensor(total_loss / max(num_batches, 1))
        if torch.distributed.is_initialized():
            loss_tensor = loss_tensor.cuda()
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
            loss_tensor = loss_tensor.cpu()
        avg_loss = loss_tensor.item()

        metrics = {f"{metric_key_prefix}_loss": avg_loss}

        # Compute recall metrics only on rank 0
        if rank == 0 and self.compute_metrics is not None:
            from transformers import EvalPrediction
            eval_pred = EvalPrediction(
                predictions=(table_all.numpy(), text_all.numpy()),
                label_ids=None,
            )
            recall_metrics = self.compute_metrics(eval_pred)
            metrics.update({
                f"{metric_key_prefix}_{k}": v for k, v in recall_metrics.items()
            })
            # Print so it's visible regardless of logging backend
            print(f"\n{'='*60}")
            print(f"[Eval] step={self.state.global_step}  N={table_all.shape[0]}")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
            print('='*60)

        # Broadcast metrics from rank 0 to all ranks so EarlyStopping works correctly
        if torch.distributed.is_initialized():
            metrics_list = [metrics] if rank == 0 else [None]
            torch.distributed.broadcast_object_list(metrics_list, src=0)
            metrics = metrics_list[0]

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics


def filter_samples_by_embedding_keys(samples: List[Dict], available_keys: Set[str], split_name: str):
    orig_len = len(samples)
    filtered = [s for s in samples if ContrastiveDataset._build_sample_key(s) in available_keys]
    print(f"   ✓ {split_name}: retained {len(filtered)} / {orig_len} samples with valid pre-computed embeddings.")
    return filtered


def _safe_table_length(sample: Dict) -> int:
    raw = sample.get("table_length", 0)
    if pd.isna(raw):
        return 0
    try:
        return int(raw)
    except Exception:
        return 0


def apply_table_length_filters(
    samples: List[Dict],
    data_args: DataArguments,
    split_name: str,
    apply_short_ratio: bool = True,
) -> List[Dict]:
    if not samples:
        return samples

    need_filter = (
        data_args.max_table_length is not None
        or data_args.sort_by_table_length
        or (apply_short_ratio and data_args.short_table_ratio is not None)
    )
    if not need_filter:
        return samples

    filtered = samples
    orig_len = len(filtered)

    if data_args.max_table_length is not None:
        max_len = int(data_args.max_table_length)
        filtered = [s for s in filtered if _safe_table_length(s) <= max_len]
        print(
            f"   ✓ {split_name}: max_table_length<={max_len} -> "
            f"{len(filtered)} / {orig_len}"
        )
        orig_len = len(filtered)

    if data_args.sort_by_table_length or (apply_short_ratio and data_args.short_table_ratio is not None):
        filtered = sorted(filtered, key=_safe_table_length)

    if apply_short_ratio and data_args.short_table_ratio is not None:
        ratio = float(data_args.short_table_ratio)
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"short_table_ratio must be in (0, 1], got {ratio}")
        keep_count = int(len(filtered) * ratio)
        if len(filtered) > 0:
            keep_count = max(1, keep_count)
        filtered = filtered[:keep_count]
        print(
            f"   ✓ {split_name}: short_table_ratio={ratio} -> keep {len(filtered)} samples"
        )

    return filtered


def load_type_vocab(type_vocab_file: str) -> Dict[str, int]:
    if not type_vocab_file or not os.path.exists(type_vocab_file):
        raise FileNotFoundError(f"type_vocab_file not found: {type_vocab_file}")

    with open(type_vocab_file, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)

    if not isinstance(raw_vocab, dict) or len(raw_vocab) == 0:
        raise ValueError(f"type_vocab_file must be a non-empty JSON object: {type_vocab_file}")

    type_vocab = {}
    for key, value in raw_vocab.items():
        type_vocab[str(key)] = int(value)

    return type_vocab


# ======================== Main ========================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgumentsCustom))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("=" * 80)
    print("Stage 1: Contrastive Learning — Table ↔ Free Text (same visit)")
    print(f"   Text embeddings (pre-computed): {data_args.free_text_embedding}")
    print(f"   Type vocab: {data_args.type_vocab_file}")

    if not data_args.free_text_embedding:
        raise ValueError("❌ --free_text_embedding is required!")
    if not data_args.type_vocab_file:
        raise ValueError("❌ --type_vocab_file is required!")

    # ------------------------------------------------------------------ #
    # 1.  Table vocab embedding cache
    # ------------------------------------------------------------------ #
    if data_args.table_text_embedding:
        embedding_map, _ = load_embedding_cache(data_args.table_text_embedding)
    else:
        raise ValueError("❌ --table_text_embedding is required for table vocab embedding lookup!")

    text_keys = list(embedding_map.keys())
    print("Pre-stacking embedding matrix...")
    text_matrix = torch.stack([embedding_map[k] for k in text_keys])
    del embedding_map
    import gc; gc.collect()
    print("  ✓ Embedding cache cleared from memory.")

    type_vocab = load_type_vocab(data_args.type_vocab_file)
    type_vocab_size = max(type_vocab.values()) + 1
    print(f"  ✓ Loaded type vocab with {len(type_vocab)} entries (embedding size={type_vocab_size}).")

    # ------------------------------------------------------------------ #
    # 2.  Table encoder
    # ------------------------------------------------------------------ #
    encoder_cfg = TableEncoderConfig(
        dim_out=model_args.projector_hidden_size,
        type_vocab_size=type_vocab_size,
    )
    batch_tabular_model = LongTableEncoder(config=encoder_cfg)

    model = ContrastiveModel(
        config=encoder_cfg,
        encoder=batch_tabular_model,
        temperature=data_args.temperature,
        embedding_matrix=text_matrix,
    )

    # ------------------------------------------------------------------ #
    # 3.  Datasets
    # ------------------------------------------------------------------ #
    print("\n📊 Loading and Splitting Datasets...")
    train_source_path = data_args.train_info_path
    train_base = MIMICIV(
        root_dir=data_args.root_dir,
        sample_info_path=train_source_path,
        lazy_mode=True,
        shuffle=False,
        only_structed_ehr=True,
    )
    train_samples = train_base.sample_info

    if len(train_samples) == 0:
        raise ValueError("No training samples found! Check train_info_path/sample_info_path and split settings.")

    # ------------------------------------------------------------------ #
    # Apply embedding-key filtering
    # ------------------------------------------------------------------ #
    print("\n📏 Applying embedding-key filtering based on pre-computed embeddings...")
    available_embeddings = torch.load(data_args.free_text_embedding, map_location='cpu', weights_only=True)
    available_keys = set(available_embeddings.keys())
    print(f"   ✓ Found {len(available_keys)} pre-computed free text embeddings.")
    del available_embeddings

    train_samples = filter_samples_by_embedding_keys(train_samples, available_keys, "Train")
    if len(train_samples) == 0:
        raise ValueError("❌ No matching samples found between sample_info sample_key and embeddings.pt keys!")
    train_samples = apply_table_length_filters(train_samples, data_args, "Train")
    if len(train_samples) == 0:
        raise ValueError("❌ No training samples left after table-length filtering.")

    # ------------------------------------------------------------------ #
    # Build Eval set (explicit validation file only)
    # ------------------------------------------------------------------ #
    eval_source_path = data_args.val_info_path
    val_base = MIMICIV(
        root_dir=data_args.root_dir,
        sample_info_path=eval_source_path,
        lazy_mode=True,
        shuffle=False,
        only_structed_ehr=True,
    )
    eval_samples = filter_samples_by_embedding_keys(val_base.sample_info, available_keys, "Val")
    if len(eval_samples) == 0:
        raise ValueError("❌ No validation samples left after embedding-key filtering. Check val_info_path and embeddings.")
    eval_samples = apply_table_length_filters(
        eval_samples,
        data_args,
        "Val",
        apply_short_ratio=False,
    )
    if len(eval_samples) == 0:
        raise ValueError("❌ No validation samples left after table-length filtering.")

    train_base.sample_info = train_samples
    if data_args.max_samples_per_split:
        train_base.sample_info = train_base.sample_info[:data_args.max_samples_per_split]

    train_dataset = ContrastiveDataset(
        train_base,
        embeddings_path=data_args.free_text_embedding,
        kept_blocks_path=data_args.kept_blocks_path,
        is_eval=False,
    )

    ccs_index = build_ccs_index(train_dataset)
    data_collator = create_collate_fn(
        train_dataset,
        ccs_index,
        data_args,
        vocab_keys=text_keys,
        type_vocab=type_vocab,
    )

    eval_dataset = None
    if eval_samples:
        val_base.sample_info = eval_samples
        if data_args.max_samples_per_split:
            val_base.sample_info = val_base.sample_info[:data_args.max_samples_per_split]
        eval_dataset = ContrastiveDataset(
            val_base,
            embeddings_path=data_args.free_text_embedding,
            kept_blocks_path=data_args.kept_blocks_path,
            is_eval=True,
        )

    def compute_metrics(eval_preds):
        preds = eval_preds.predictions
        # Transformers might pass tuple or dict based on version
        if isinstance(preds, tuple):
            table_embs, text_embs = torch.tensor(preds[0]), torch.tensor(preds[1])
        elif isinstance(preds, dict):
            table_embs, text_embs = torch.tensor(preds['table_embs']), torch.tensor(preds['text_embs'])
        else:
            table_embs, text_embs = torch.tensor(preds), torch.tensor(preds)

        # compute Table->Text recall
        # shape: (N_val, dim) x (dim, N_val) -> (N_val, N_val)
        sim_matrix = torch.matmul(table_embs, text_embs.t())
        N = sim_matrix.size(0)
        labels = torch.arange(N, device=sim_matrix.device)
        
        sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
        matches = (sorted_indices == labels.unsqueeze(1))
        
        metrics = {}
        for k in [1, 5, 10, 50]:
            if k <= N:
                correct_k = matches[:, :k].sum(dim=1).float().mean().item()
                metrics[f"recall@{k}"] = correct_k
                
        ranks = matches.nonzero()[:, 1] + 1
        mrr = (1.0 / ranks.float()).mean().item()
        metrics["mrr"] = mrr
        
        return metrics

    callbacks = []
    if training_args.early_stopping_patience > 0:
        if eval_dataset is None:
            print("⚠️  early_stopping_patience > 0 but no eval_dataset — early stopping disabled.")
        else:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            ))
            print(f"   ✓ Early stopping enabled: patience={training_args.early_stopping_patience}, "
                  f"metric='{training_args.metric_for_best_model}'")

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None,
        callbacks=callbacks if callbacks else None,
    )

    print("🚀 Starting Training...")
    trainer.train()
    trainer.save_model()

    print("\n💾 Saving standalone TabularEncoder")
    tabular_save_path = os.path.join(training_args.output_dir, "tabular_encoder")
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    os.makedirs(tabular_save_path, exist_ok=True)
    encoder_state_dict = {
        key: value.detach().cpu()
        for key, value in model_to_save.encoder.state_dict().items()
    }
    save_file(
        encoder_state_dict,
        os.path.join(tabular_save_path, "model.safetensors")
    )
    print(f"   ✓ Saved to {tabular_save_path}")


if __name__ == "__main__":
    main()
