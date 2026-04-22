import os
import sys
import argparse
import subprocess
import time
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from dataset.mimic.mimic_dataset import MIMICIV
import glob


def _in_distributed_env():
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        return True
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except Exception:
        return False


def _maybe_auto_launch_distributed(args):
    """
    If user runs this script via plain `python` on a multi-GPU machine,
    auto relaunch via torchrun so all visible GPUs are used.
    """
    if args.disable_auto_launch:
        return
    if _in_distributed_env():
        return
    if not torch.cuda.is_available():
        return

    visible_gpus = torch.cuda.device_count()
    if visible_gpus <= 1:
        return

    requested = args.auto_num_processes if args.auto_num_processes and args.auto_num_processes > 0 else visible_gpus
    num_procs = max(1, min(requested, visible_gpus))
    if num_procs <= 1:
        return

    script_path = os.path.abspath(__file__)
    forward_args = sys.argv[1:] + ["--disable_auto_launch"]
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_procs}",
        script_path,
        *forward_args,
    ]
    print(f"🚀 Auto-launching distributed inference with {num_procs} GPUs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    sys.exit(0)


def save_embeddings_chunk(save_dir, rank, embeddings_dict):
    """Save a chunk of embeddings for a specific rank"""
    save_path = os.path.join(save_dir, f"embeddings_rank_{rank}.pt")
    torch.save(embeddings_dict, save_path)
    print(f"   💾 Saved rank {rank} chunk with {len(embeddings_dict)} samples to {save_path}")


def build_sample_key(sample_info):
    return (
        f"{sample_info.get('subject_id', '')}|"
        f"{sample_info.get('task', '')}|"
        f"{sample_info.get('context_begin', '')}|"
        f"{sample_info.get('context_end', '')}"
    )


def load_processed_keys_from_part_files(part_files, rank_for_log=None):
    processed_keys = set()
    for pf in part_files:
        try:
            data = torch.load(pf, map_location='cpu', weights_only=True)
            processed_keys.update(data.keys())
        except Exception as e:
            prefix = f"[Rank {rank_for_log}] " if rank_for_log is not None else ""
            print(f"⚠️ {prefix}Corrupted checkpoint {pf}: {e}")
    return processed_keys


def build_resume_meta(part_files):
    if not part_files:
        return {"num_part_files": 0, "max_part_mtime": 0.0}
    max_part_mtime = max(os.path.getmtime(p) for p in part_files)
    return {"num_part_files": len(part_files), "max_part_mtime": float(max_part_mtime)}


def resume_cache_matches(cache_data, resume_meta):
    if not isinstance(cache_data, dict):
        return False
    try:
        cache_num = int(cache_data.get("num_part_files", -1))
        cache_mtime = float(cache_data.get("max_part_mtime", -1.0))
    except Exception:
        return False
    return (
        cache_num == int(resume_meta["num_part_files"])
        and cache_mtime >= float(resume_meta["max_part_mtime"]) - 1e-6
    )


def infer_kept_block_ids(blocks, tokenizer, max_length, content_budget_tokens):
    """
    Infer which text blocks survive left-side truncation under a token budget.
    We keep a suffix of blocks whose tokenized length fits into content_budget_tokens.
    """
    if not blocks:
        return []

    block_texts = [str(b.get("text", "")) for b in blocks]
    block_ids = [str(b.get("block_id", "")) for b in blocks]
    if content_budget_tokens <= 0:
        return []

    enc = tokenizer(block_texts, add_special_tokens=False, truncation=False)
    block_lens = [len(ids) for ids in enc["input_ids"]]
    sep_len = len(tokenizer("\n\n", add_special_tokens=False, truncation=False)["input_ids"])

    kept = []
    used = 0
    for i in range(len(blocks) - 1, -1, -1):
        blk_len = block_lens[i]
        extra_sep = sep_len if kept else 0
        if used + blk_len + extra_sep > content_budget_tokens:
            continue
        kept.append(block_ids[i])
        used += blk_len + extra_sep

    kept.reverse()
    return kept


def safe_destroy_process_group():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as e:
            print(f"⚠️ Failed to destroy process group cleanly: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/ma-user/modelarts/user-job-dir/LiverTransplantation/model_weights/BlueZeros/EHR-R1-1.7B")
    parser.add_argument("--root_dir", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular")
    parser.add_argument("--sample_csv", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/train_val/contrastive_learning.csv")
    parser.add_argument("--output_dir", type=str, default="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/embeddings/table_free_text")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=32000)
    parser.add_argument("--save_steps", type=int, default=100, help="Save processing checkpoint every N batches")
    parser.add_argument("--max_table_length", type=int, default=None, help="Maximum table length to keep a sample")
    parser.add_argument("--sort_by_table_length", action="store_true", help="Sort the dataset by table_length (ascending)")
    parser.add_argument("--short_table_ratio", type=float, default=None, help="Ratio of shortest table samples to keep")
    parser.add_argument("--auto_num_processes", type=int, default=0, help="Auto-launch GPU processes when run with plain python. 0 means all visible GPUs.")
    parser.add_argument("--disable_auto_launch", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--global_resume_cache_wait_seconds", type=int, default=7200, help="How long non-main ranks wait for valid global resume cache before fallback scan.")
    parser.add_argument("--global_resume_cache_poll_seconds", type=float, default=2.0, help="Polling interval while waiting for global resume cache.")
    parser.add_argument("--force_rebuild_global_resume_cache", action="store_true", help="Ignore existing _global_processed_keys.pt and rebuild from shard files.")
    parser.add_argument("--merge_wait_seconds", type=int, default=7200, help="How long rank0 waits for all rank done markers before merge.")
    parser.add_argument("--merge_poll_seconds", type=float, default=2.0, help="Polling interval while waiting for rank done markers.")
    args = parser.parse_args()

    _maybe_auto_launch_distributed(args)

    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Setup output
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    
    if accelerator.is_main_process:
        print(f"🚀 Saving embeddings to: {save_dir}")
        print(f"🌍 Distributed Mode: {accelerator.num_processes} GPUs detected.")
        # Remove stale done markers from prior interrupted runs.
        for marker in glob.glob(os.path.join(save_dir, "rank_*.done")):
            try:
                os.remove(marker)
            except OSError:
                pass
    
    accelerator.wait_for_everyone()

    # 1. Load Model & Tokenizer
    if accelerator.is_main_process:
        print("📦 Loading LLM...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": accelerator.device} # Map to current device
    ).eval()
    
    # 2. Load Dataset
    if accelerator.is_main_process:
        print("📊 Loading Dataset...")
        
    # only_structed_ehr=True returns:
    # - sample['input']: table-aligned markdown-like text
    # - sample['structured_text_blocks']: block list used to build the input text
    # - sample['measurement_table_row_block_ids']: row->block alignment metadata
    dataset = MIMICIV(
        root_dir=args.root_dir,
        sample_info_path=args.sample_csv,
        lazy_mode=True,
        shuffle=False,
        only_structed_ehr=True,
    )
    
    # Tag original indices to properly map to the embeddings cache dict later,
    # because filtering changes the array positions!
    for i, s in enumerate(dataset.sample_info):
        s['original_index'] = i

    if args.max_table_length is not None or args.sort_by_table_length or args.short_table_ratio is not None:
        if accelerator.is_main_process:
            print("\n📏 Applying table length filtering / sorting...")
        
        has_table_length = all('table_length' in s for s in dataset.sample_info[:10])
        if not has_table_length:
            if accelerator.is_main_process:
                print("   ⚠️ WARNING: 'table_length' column not found in sample_info. Filtering skipped.")
        else:
            orig_len = len(dataset.sample_info)
            if args.max_table_length is not None:
                dataset.sample_info = [s for s in dataset.sample_info if s.get('table_length', 0) <= args.max_table_length]
            
            if args.sort_by_table_length or args.short_table_ratio is not None:
                dataset.sample_info = sorted(dataset.sample_info, key=lambda s: s.get('table_length', 0))
                
                if args.short_table_ratio is not None:
                    keep_count = int(len(dataset.sample_info) * args.short_table_ratio)
                    dataset.sample_info = dataset.sample_info[:keep_count]
            
            if accelerator.is_main_process:
                print(f"   ✓ Filtered table samples: {orig_len} -> {len(dataset.sample_info)}")
    
    all_indices = list(range(len(dataset.sample_info)))

    # --- Global Resume Logic ---
    rank_part_pattern = os.path.join(save_dir, f"rank_{accelerator.process_index}_part_*.pt")
    existing_rank_parts = sorted(glob.glob(rank_part_pattern))
    part_counter = len(existing_rank_parts)

    global_resume_cache = os.path.join(save_dir, "_global_processed_keys.pt")
    global_processed_keys = set()
    # Only embedding shards; avoid matching kept-block shard files.
    all_part_files = sorted(glob.glob(os.path.join(save_dir, "rank_[0-9]*_part_*.pt")))
    resume_meta = build_resume_meta(all_part_files)

    if accelerator.is_main_process:
        loaded_from_valid_cache = False
        if (not args.force_rebuild_global_resume_cache) and os.path.exists(global_resume_cache):
            try:
                cache_data = torch.load(global_resume_cache, map_location='cpu')
                if resume_cache_matches(cache_data, resume_meta):
                    global_processed_keys = set(cache_data.get("processed_keys", []))
                    loaded_from_valid_cache = True
                    print(
                        f"   ♻️ Reusing global resume cache "
                        f"({len(global_processed_keys)} processed keys, {resume_meta['num_part_files']} shard files)."
                    )
            except Exception as e:
                print(f"   ⚠️ Failed to load existing global resume cache: {e}")

        if not loaded_from_valid_cache:
            print(
                f"   🔍 Building global resume cache from {resume_meta['num_part_files']} shard files..."
            )
            global_processed_keys = load_processed_keys_from_part_files(all_part_files)
            torch.save(
                {
                    "processed_keys": list(global_processed_keys),
                    "num_part_files": int(resume_meta["num_part_files"]),
                    "max_part_mtime": float(resume_meta["max_part_mtime"]),
                    "updated_at": time.time(),
                },
                global_resume_cache,
            )

    if not accelerator.is_main_process:
        loaded_valid_cache = False
        deadline = time.time() + max(5, int(args.global_resume_cache_wait_seconds))
        poll_seconds = max(0.1, float(args.global_resume_cache_poll_seconds))
        while time.time() < deadline:
            if os.path.exists(global_resume_cache):
                try:
                    cache_data = torch.load(global_resume_cache, map_location='cpu')
                    if resume_cache_matches(cache_data, resume_meta):
                        global_processed_keys = set(cache_data.get("processed_keys", []))
                        loaded_valid_cache = True
                        break
                except Exception:
                    pass
            time.sleep(poll_seconds)

        if not loaded_valid_cache:
            # Fallback if no valid cache is available in time.
            print(
                f"⚠️ [Rank {accelerator.process_index}] Timed out waiting for valid global resume cache. "
                "Falling back to local shard scan."
            )
            global_processed_keys = load_processed_keys_from_part_files(
                all_part_files,
                rank_for_log=accelerator.process_index,
            )

    remaining_indices = [
        i for i in all_indices
        if build_sample_key(dataset.sample_info[i]) not in global_processed_keys
    ]
    my_indices = remaining_indices[accelerator.process_index::accelerator.num_processes]

    if accelerator.is_main_process:
        print(f"   Total samples: {len(all_indices)}")
        print(f"   Globally processed samples: {len(global_processed_keys)}")
        print(f"   Remaining samples (global): {len(remaining_indices)}")
        print(f"   Samples per GPU (remaining): ~{len(my_indices)}")

    print(
        f"   [Rank {accelerator.process_index}] Resuming with global state: "
        f"{len(my_indices)} assigned samples, existing rank parts={len(existing_rank_parts)}."
    )
    # -------------------------
    
    # 3. Processing Loop
    local_embeddings = {} # Buffer for this process
    local_kept_blocks = {}
    batch_texts = []
    batch_keys = []
    batch_kept_blocks = []
    
    # Use tqdm only on main process or with distinct position
    # Fix tqdm total to reflect filtered indices
    iterator = tqdm(my_indices, disable=not accelerator.is_main_process, desc="Generating")
    
    batches_since_save = 0
    
    for idx in iterator:
        sample = dataset[idx]
        content_text = sample["input"]
        sample_key = build_sample_key(dataset.sample_info[idx])
        
        text = content_text
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": text}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            empty_messages = [{"role": "user", "content": ""}]
            template_empty = tokenizer.apply_chat_template(empty_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            template_overhead = len(tokenizer(template_empty, add_special_tokens=False, truncation=False)["input_ids"])
        else:
            template_overhead = 0

        batch_texts.append(text)
        batch_keys.append(sample_key)

        content_budget = max(0, args.max_length - template_overhead)
        kept_block_ids = infer_kept_block_ids(
            sample.get("structured_text_blocks", []),
            tokenizer,
            args.max_length,
            content_budget,
        )
        batch_kept_blocks.append(kept_block_ids)
        
        if len(batch_texts) >= args.batch_size or idx == my_indices[-1]:
            # Encode batch
            inputs = tokenizer(
                batch_texts, 
                max_length=args.max_length, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
            
            # Strict EOS Pooling
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            model_name_lower = args.model_path.lower()
            eos_id = None
            if "qwen" in model_name_lower or "ehr-r1" in model_name_lower:
                chatml_end = "<|im_end|>"
                eos_id = tokenizer.convert_tokens_to_ids(chatml_end)

            # Find indices
            last_indices = None
            if eos_id is not None:
                eos_mask = (input_ids == eos_id)
                if eos_mask.sum().item() > 0:
                    seq_len = input_ids.size(1)
                    indices = torch.arange(seq_len, device=input_ids.device)
                    last_indices = (eos_mask * indices).argmax(dim=1)
            if last_indices is None:
                last_indices = attention_mask.sum(dim=1) - 1
            
            # Gather
            batch_size = last_hidden.size(0)
            pooled = last_hidden[torch.arange(batch_size, device=last_hidden.device), last_indices]
            
            # Store in buffer
            for i, key in enumerate(batch_keys):
                # (Hidden,)
                hs = pooled[i].squeeze().cpu().to(torch.bfloat16)
                local_embeddings[key] = hs
                local_kept_blocks[key] = batch_kept_blocks[i]
            
            batch_texts = []
            batch_keys = []
            batch_kept_blocks = []
            batches_since_save += 1

            # Checkpoint
            if batches_since_save >= args.save_steps:
                embed_path = os.path.join(save_dir, f"rank_{accelerator.process_index}_part_{part_counter}.pt")
                kept_blocks_path = os.path.join(save_dir, f"rank_{accelerator.process_index}_kept_blocks_part_{part_counter}.pt")
                torch.save(local_embeddings, embed_path)
                torch.save(local_kept_blocks, kept_blocks_path)
                print(f"   [Rank {accelerator.process_index}] 💾 Saved checkpoint part {part_counter} ({len(local_embeddings)} samples)")
                local_embeddings.clear()
                local_kept_blocks.clear()
                batches_since_save = 0
                part_counter += 1

    # 4. Save remaining buffer
    if local_embeddings:
        embed_path = os.path.join(save_dir, f"rank_{accelerator.process_index}_part_{part_counter}.pt")
        kept_blocks_path = os.path.join(save_dir, f"rank_{accelerator.process_index}_kept_blocks_part_{part_counter}.pt")
        torch.save(local_embeddings, embed_path)
        torch.save(local_kept_blocks, kept_blocks_path)
        print(f"   [Rank {accelerator.process_index}] 💾 Saved final part {part_counter} ({len(local_embeddings)} samples)")

    # Mark this rank as done, then tear down process group before long merge.
    done_marker = os.path.join(save_dir, f"rank_{accelerator.process_index}.done")
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write(str(time.time()))
    safe_destroy_process_group()

    # 5. Merge (Main process only)
    if accelerator.is_main_process:
        # Wait for all rank done markers via filesystem polling to avoid NCCL timeout.
        wait_deadline = time.time() + max(5, int(args.merge_wait_seconds))
        poll_seconds = max(0.1, float(args.merge_poll_seconds))
        expected_markers = accelerator.num_processes
        while time.time() < wait_deadline:
            marker_count = len(glob.glob(os.path.join(save_dir, "rank_*.done")))
            if marker_count >= expected_markers:
                break
            time.sleep(poll_seconds)

        marker_count = len(glob.glob(os.path.join(save_dir, "rank_*.done")))
        if marker_count < expected_markers:
            print(
                f"⚠️ Merge starting with {marker_count}/{expected_markers} done markers "
                f"after waiting {args.merge_wait_seconds}s."
            )

        print("\n🔄 Merging all chunks into single file...")
        full_embeddings = {}
        full_kept_blocks = {}
        
        # Keep patterns disjoint:
        # - embedding shards: rank_<int>_part_<int>.pt
        # - kept-block shards: rank_<int>_kept_blocks_part_<int>.pt
        chunk_files = sorted(glob.glob(os.path.join(save_dir, "rank_[0-9]*_part_*.pt")))
        kept_chunk_files = sorted(glob.glob(os.path.join(save_dir, "rank_[0-9]*_kept_blocks_part_*.pt")))
        print(f"   Found {len(chunk_files)} chunk files to merge.")
        
        for chunk_path in tqdm(chunk_files, desc="Merging"):
            try:
                chunk_data = torch.load(chunk_path, weights_only=True)
                full_embeddings.update(chunk_data)
            except Exception as e:
                print(f"⚠️ Error loading {chunk_path}: {e}")

        for chunk_path in tqdm(kept_chunk_files, desc="Merging kept blocks"):
            try:
                chunk_data = torch.load(chunk_path, weights_only=True)
                full_kept_blocks.update(chunk_data)
            except Exception as e:
                print(f"⚠️ Error loading {chunk_path}: {e}")

        final_path = os.path.join(save_dir, "embeddings.pt")
        kept_blocks_final_path = os.path.join(save_dir, "kept_blocks.pt")
        torch.save(full_embeddings, final_path)
        torch.save(full_kept_blocks, kept_blocks_final_path)
        print(f"✅ Saved merged embeddings to {final_path}")
        print(f"✅ Saved merged kept blocks to {kept_blocks_final_path}")
        print(f"   Total samples: {len(full_embeddings)}")

if __name__ == "__main__":
    main()
