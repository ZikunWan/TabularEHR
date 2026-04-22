"""
Generate pre-computed PubMedBERT embeddings for all unique table cell texts in MIMIC-IV.
This creates the embedding cache for Item names, Unit names, and non-numeric Value texts.

The output is used by contrastive_learning.py via --embedding_cache_path.

Supports resume: Phase 1 caches unique texts to JSON; Phase 2 saves partial .pt files.

Usage (4 GPUs):
    accelerate launch --num_processes 4 preprocess/mimic_iv/8_generate_table_embeddings.py

Single GPU:
    python preprocess/mimic_iv/8_generate_table_embeddings.py [same args]
"""
import os
import sys
import json
import glob
import time
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from transformers import AutoModel, AutoTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dataset.mimic_dataset import MIMICIV

# ---- Per-process global dataset (created once per worker via initializer) ----
_worker_dataset = None

def _init_worker(root_dir, sample_csv):
    """Each subprocess creates its own MIMICIV instance."""
    global _worker_dataset
    _worker_dataset = MIMICIV(
        root_dir=root_dir,
        sample_info_path=sample_csv,
        lazy_mode=True,
        shuffle=False,
        return_table=True,
        log=False,
    )

def _scan_one_sample(idx):
    """Extract unique texts from a single sample. Runs in subprocess."""
    try:
        sample = _worker_dataset[idx]
    except Exception:
        return set()

    tables = sample.get('measurement_table')
    if tables is None:
        return set()

    if isinstance(tables, pd.DataFrame) and len(tables) > 0:
        df = tables
    elif isinstance(tables, dict):
        frames = [v for v in tables.values() if isinstance(v, pd.DataFrame) and len(v) > 0]
        if not frames:
            return set()
        df = pd.concat(frames, ignore_index=True)
    else:
        return set()

    texts = set()
    if 'Item' in df.columns:
        texts.update(df['Item'].astype(str).tolist())
    if 'Value' in df.columns:
        for val in df['Value'].astype(str).tolist():
            try:
                float(val)
            except (ValueError, TypeError):
                texts.add(val)
    if 'Unit' in df.columns:
        texts.update(df['Unit'].astype(str).fillna('-').tolist())
    return texts


def run_phase1(args, save_dir, texts_cache_path):
    """
    Phase 1: Collect unique texts. Runs BEFORE Accelerator init.
    Only rank 0 scans; other ranks poll-wait for the cache file.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if os.path.exists(texts_cache_path):
        if local_rank == 0:
            print(f"\n✅ Phase 1 cache found: {texts_cache_path}, skipping scan.")
        return  # All ranks return immediately

    if local_rank == 0:
        # ---- Rank 0: scan and save ----
        os.makedirs(save_dir, exist_ok=True)
        print("\n" + "=" * 60)
        print("Phase 1: Collecting unique texts from MIMIC-IV tables...")
        print(f"  Using {args.num_workers} CPU processes (multiprocessing)")
        print("=" * 60)

        tmp_dataset = MIMICIV(
            root_dir=args.root_dir,
            sample_info_path=args.sample_csv,
            lazy_mode=True, shuffle=False, return_table=True, log=False,
        )
        num_samples = len(tmp_dataset)
        if args.max_samples:
            num_samples = min(num_samples, args.max_samples)
        del tmp_dataset
        print(f"Processing {num_samples} samples...")

        with Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(args.root_dir, args.sample_csv),
        ) as pool:
            results = list(tqdm(
                pool.imap(_scan_one_sample, range(num_samples)),
                total=num_samples, desc="Scanning tables"
            ))

        text_set = set()
        for r in results:
            text_set.update(r)
        for tok in ['[PAD]', '[EMPTY]', '-', '0', '[MASK]']:
            text_set.add(tok)

        unique_texts = sorted(text_set)
        print(f"\nTotal unique texts: {len(unique_texts)}")

        with open(texts_cache_path, 'w', encoding='utf-8') as f:
            json.dump(unique_texts, f, ensure_ascii=False)
        print(f"💾 Saved Phase 1 cache to {texts_cache_path}")

    else:
        # ---- Other ranks: poll-wait for cache file ----
        print(f"[Rank {local_rank}] Waiting for Phase 1 cache from rank 0...")
        while not os.path.exists(texts_cache_path):
            time.sleep(5)
        # Small delay to ensure file is fully written
        time.sleep(2)
        print(f"[Rank {local_rank}] Phase 1 cache found, proceeding.")


def main():
    parser = argparse.ArgumentParser(description="Generate PubMedBERT embeddings for MIMIC-IV table texts")
    parser.add_argument("--bert_model", type=str,
                        default="/home/ma-user/sfs_turbo/sai6/zkwan/model_weights/PubMedBERT")
    parser.add_argument("--root_dir", type=str,
                        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular")
    parser.add_argument("--sample_csv", type=str,
                        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/all/contrastive_learning.csv")
    parser.add_argument("--output_path", type=str,
                        default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/embeddings/table_text_embeddings.pt")
    parser.add_argument("--batch_size", type=int, default=64, help="BERT encoding batch size per GPU")
    parser.add_argument("--max_token_len", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for debugging")
    parser.add_argument("--num_workers", type=int, default=64, help="CPU workers for scanning")
    parser.add_argument("--save_every", type=int, default=200, help="Save checkpoint every N batches")
    args = parser.parse_args()

    save_dir = os.path.dirname(args.output_path)
    texts_cache_path = os.path.join(save_dir, "_unique_texts.json")

    # ============================================================
    # Phase 1: BEFORE Accelerator (no NCCL timeout issue)
    # ============================================================
    run_phase1(args, save_dir, texts_cache_path)

    # ============================================================
    # Phase 2: Encode with BERT (distributed across GPUs)
    # Now safe to init Accelerator — Phase 1 is done on all ranks
    # ============================================================
    from accelerate import Accelerator
    from accelerate.utils import gather_object

    accelerator = Accelerator()
    rank = accelerator.process_index
    is_main = accelerator.is_main_process

    # Load unique texts from cache (all ranks)
    with open(texts_cache_path, 'r', encoding='utf-8') as f:
        unique_texts = json.load(f)

    if is_main:
        print(f"\n🌍 Using {accelerator.num_processes} GPU(s)")
        print(f"BERT Model: {args.bert_model}")
        print(f"Output: {args.output_path}")
        print(f"Unique texts: {len(unique_texts)}")
        print("\n" + "=" * 60)
        print("Phase 2: Encoding texts with BERT...")
        print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    model = AutoModel.from_pretrained(args.bert_model).to(accelerator.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    text_dim = model.config.hidden_size

    # Shard texts across GPUs
    my_texts = unique_texts[rank::accelerator.num_processes]

    # --- Resume: load existing partial results for this rank ---
    local_embeddings = {}
    part_pattern = os.path.join(save_dir, f"rank_{rank}_part_*.pt")
    existing_parts = sorted(glob.glob(part_pattern))
    part_counter = 0

    for pf in existing_parts:
        try:
            data = torch.load(pf, map_location='cpu', weights_only=True)
            local_embeddings.update(data)
            part_counter += 1
        except Exception as e:
            print(f"⚠️ [Rank {rank}] Corrupted checkpoint {pf}: {e}")

    if local_embeddings:
        print(f"[Rank {rank}] Resumed {len(local_embeddings)} already-encoded texts from {part_counter} parts")
        my_texts = [t for t in my_texts if t not in local_embeddings]
        print(f"[Rank {rank}] {len(my_texts)} texts remaining")

    if is_main:
        print(f"Text dim: {text_dim}, ~{len(my_texts)} texts per GPU (after resume)")

    # --- Encode ---
    buffer = {}
    batches_since_save = 0

    for i in tqdm(range(0, len(my_texts), args.batch_size),
                  desc=f"GPU {rank}", disable=not is_main):
        batch_texts = my_texts[i:i + args.batch_size]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_token_len,
            return_tensors='pt'
        ).to(accelerator.device)

        with torch.no_grad():
            out = model(**tokens)
            embs = out.last_hidden_state[:, 0, :]  # CLS token

        for j, text in enumerate(batch_texts):
            buffer[text] = embs[j].cpu()

        batches_since_save += 1
        if batches_since_save >= args.save_every:
            save_path = os.path.join(save_dir, f"rank_{rank}_part_{part_counter}.pt")
            local_embeddings.update(buffer)
            torch.save(local_embeddings, save_path)
            if is_main:
                print(f"\n💾 Checkpoint: {len(local_embeddings)} texts saved (part {part_counter})")
            buffer.clear()
            batches_since_save = 0
            part_counter += 1

    # Save remaining buffer
    if buffer:
        local_embeddings.update(buffer)
        save_path = os.path.join(save_dir, f"rank_{rank}_part_{part_counter}.pt")
        torch.save(local_embeddings, save_path)

    # ============================================================
    # Phase 3: Gather and merge
    # ============================================================
    accelerator.wait_for_everyone()

    all_dicts = gather_object([local_embeddings])

    if is_main:
        print(f"\n🔄 Merging results from {len(all_dicts)} GPU(s)...")
        embeddings_dict = {}
        for d in all_dicts:
            embeddings_dict.update(d)

        print(f"Total encoded: {len(embeddings_dict)} unique texts")

        torch.save({
            'embeddings': embeddings_dict,
            'text_dim': text_dim,
            'bert_model': args.bert_model,
        }, args.output_path)

        file_size = os.path.getsize(args.output_path) / (1024 * 1024)
        print(f"Cache file size: {file_size:.2f} MB")

        # Cleanup partial files
        for pf in glob.glob(os.path.join(save_dir, "rank_*_part_*.pt")):
            os.remove(pf)
        if os.path.exists(texts_cache_path):
            os.remove(texts_cache_path)
        print("🧹 Cleaned up checkpoint files")
        print("✅ Done!")


if __name__ == "__main__":
    main()
