"""
Generate pre-computed BERT embeddings for all unique texts in the MIMIC-IV-CDM dataset.
Run this BEFORE training to create the embedding cache.

Single-GPU usage:
    python preprocess/mimic_iv_cdm/4_generate_embeddings.py

Multi-GPU usage (faster):
    torchrun --nproc_per_node=4 preprocess/mimic_iv_cdm/4_generate_embeddings.py
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dataset.mimic_iv_cdm_dataset import MIMICIVCDM


# ── Configuration ──────────────────────────────────────────────────────────────
BERT_MODEL  = "/home/ma-user/sfs_turbo/sai6/zkwan/model_weights/PubMedBERT"
DATA_DIR    = "/home/ma-user/sfs_turbo/Data/mimic-iv-cdm"
CACHE_DIR   = "/home/ma-user/sfs_turbo/sai6/zkwan/.cache/embeddings/mimic_iv_cdm"
OUTPUT_PATH = os.path.join(CACHE_DIR, "text_embeddings.pt")
TASK_NAME   = "MIMIC-IV-CDM Main Disease Diagnoses"
BATCH_SIZE  = 512
MAX_TOKEN_LEN = 512   # MIMIC-IV-CDM item/unit/value texts are short


def is_distributed():
    return "LOCAL_RANK" in os.environ


def collect_unique_texts(all_splits=("train", "val", "test")) -> list:
    """Load all splits and harvest every unique Item / Value / Unit string."""
    unique_texts = set()

    for split in all_splits:
        try:
            ds = MIMICIVCDM(
                root_dir=DATA_DIR,
                split=split,
                task_name=TASK_NAME,
                return_table=True,
                lazy_mode=False,
                shuffle=False,
            )
        except Exception as e:
            print(f"  Skipping split '{split}': {e}")
            continue

        print(f"  [{split}] {len(ds)} samples")
        for i in tqdm(range(len(ds)), desc=f"  Harvesting {split}"):
            sample = ds[i]
            df = sample.get("measurement_table")
            if df is None or df.empty:
                continue

            # Item names
            for text in df["Item"].astype(str).tolist():
                unique_texts.add(text)

            # Values — keep non-numeric strings only (numeric ones use '0' embedding)
            for val in df["Value"].astype(str).tolist():
                try:
                    float(val)
                except ValueError:
                    unique_texts.add(val)

            # Units
            if "Unit" in df.columns:
                for unit in df["Unit"].astype(str).fillna("-").tolist():
                    unique_texts.add(unit)

    # Special tokens always required by collate_fn
    for special in ("[PAD]", "[EMPTY]", "-", "0"):
        unique_texts.add(special)

    return list(unique_texts)


def encode_texts(texts: list, model, tokenizer, device, batch_size: int) -> dict:
    """Encode a list of strings → dict {text: cpu_tensor}."""
    embeddings = {}
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**tokens)
            embs = out.last_hidden_state[:, 0, :].cpu()  # CLS token
        for j, text in enumerate(batch):
            embeddings[text] = embs[j]
    return embeddings


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Distributed setup ──────────────────────────────────────────────────────
    if is_distributed():
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.device(f"cuda:{rank}")
    else:
        rank       = 0
        world_size = 1
        device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Device(s): {world_size}x GPU" if world_size > 1 else f"Device: {device}")
        print(f"BERT Model: {BERT_MODEL}")
        print(f"Output: {OUTPUT_PATH}\n")

    # ── Step 1: Collect unique texts (rank 0 only, then broadcast) ─────────────
    if rank == 0:
        print("="*55)
        print("Step 1 — Harvesting unique texts from all splits")
        print("="*55)
        unique_texts = collect_unique_texts()
        print(f"\nTotal unique texts: {len(unique_texts)}")
    else:
        unique_texts = None

    if is_distributed():
        import torch.distributed as dist
        # Broadcast list length, then the list itself via object list
        obj = [unique_texts]
        dist.broadcast_object_list(obj, src=0)
        unique_texts = obj[0]

    # ── Step 2: Load BERT ──────────────────────────────────────────────────────
    if rank == 0:
        print("\n" + "="*55)
        print("Step 2 — Loading BERT model")
        print("="*55)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    text_dim = model.config.hidden_size
    if rank == 0:
        print(f"Embedding dimension: {text_dim}")

    # ── Step 3: Encode (sharded across GPUs) ──────────────────────────────────
    if rank == 0:
        print("\n" + "="*55)
        print("Step 3 — Encoding")
        print("="*55)

    my_shard = np.array_split(unique_texts, world_size)[rank]
    if rank == 0:
        print(f"Encoding {len(my_shard)} texts on rank 0 (total {len(unique_texts)})...")

    local_embeddings = encode_texts(list(my_shard), model, tokenizer, device, BATCH_SIZE)

    # ── Step 4: Merge and save (rank 0) ───────────────────────────────────────
    if is_distributed():
        import torch.distributed as dist
        # Each rank saves a partial file; rank 0 merges
        partial_path = os.path.join(CACHE_DIR, f"partial_embs_rank_{rank}.pt")
        torch.save(local_embeddings, partial_path)
        dist.barrier()

        if rank == 0:
            print("\nMerging partial results...")
            final_embeddings = {}
            for r in range(world_size):
                p = os.path.join(CACHE_DIR, f"partial_embs_rank_{r}.pt")
                final_embeddings.update(torch.load(p, weights_only=False))
                os.remove(p)
        dist.destroy_process_group()
    else:
        final_embeddings = local_embeddings

    if rank == 0:
        print(f"\nSaving {len(final_embeddings)} embeddings to {OUTPUT_PATH}...")
        torch.save({
            "embeddings": final_embeddings,
            "text_dim":   text_dim,
            "bert_model": BERT_MODEL,
        }, OUTPUT_PATH)
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"Cache size: {size_mb:.1f} MB")
        print("Done!")


if __name__ == "__main__":
    main()
