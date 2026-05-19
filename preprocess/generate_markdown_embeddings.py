"""
Generate markdown/free-text embeddings used as contrastive positives.

Examples:
    python preprocess/generate_markdown_embeddings.py \
        --dataset eicu \
        --root_dir /data/zikun_workspace/eicu-crd \
        --processed_dir /data/zikun_workspace/eicu-crd/processed \
        --sample_info_path /data/zikun_workspace/eicu-crd/processed/sample_info_train.json \
        --output_dir /data/zikun_workspace/eicu-crd/embeddings/table_free_text

    python preprocess/generate_markdown_embeddings.py \
        --dataset ehrshot \
        --root_dir /data/EHR_data_public/EHRSHOT \
        --sample_info_path /data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv \
        --output_dir /data/EHR_data_public/EHRSHOT/embeddings/table_free_text
"""

import argparse
import glob
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.mimic.mimic_dataset import MIMICIV


def in_distributed_env():
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        return True
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def maybe_auto_launch_distributed(args):
    if args.disable_auto_launch or in_distributed_env() or not torch.cuda.is_available():
        return
    visible_gpus = torch.cuda.device_count()
    requested = args.auto_num_processes if args.auto_num_processes > 0 else visible_gpus
    num_procs = max(1, min(requested, visible_gpus))
    if num_procs <= 1:
        return

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_procs}",
        os.path.abspath(__file__),
        *sys.argv[1:],
        "--disable_auto_launch",
    ]
    print(f"Auto-launching distributed inference with {num_procs} GPUs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    sys.exit(0)


def safe_destroy_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def build_sample_key(dataset_name, sample_info):
    if "sample_id" in sample_info:
        return str(sample_info["sample_id"])
    if dataset_name == "mimic_iv":
        return (
            "mimic_iv|"
            f"{sample_info.get('subject_id', '')}|"
            f"{sample_info.get('task', '')}|"
            f"{sample_info.get('context_begin', '')}|"
            f"{sample_info.get('context_end', '')}"
        )
    if dataset_name == "eicu":
        return (
            "eicu|"
            f"{sample_info.get('patient_id', '')}|"
            f"{sample_info.get('icustay_id', '')}|"
            f"{sample_info.get('task_name', '')}|"
            f"{sample_info.get('obs_hours', '')}|"
            f"{sample_info.get('gap_hours', '')}|"
            f"{sample_info.get('pred_hours', '')}"
        )
    if dataset_name == "ehrshot":
        return (
            "ehrshot|"
            f"{sample_info.get('patient_id', '')}|"
            f"{sample_info.get('task_name', '')}|"
            f"{sample_info.get('prediction_time', '')}"
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_dataset(args):
    if args.dataset == "mimic_iv":
        dataset = MIMICIV(
            root_dir=args.root_dir,
            sample_info_path=args.sample_info_path[0],
            lazy_mode=True,
            shuffle=False,
            table_mode="table_only",
            use_table_length_cache=False,
        )
        for sample_info_path in args.sample_info_path[1:]:
            extra_dataset = MIMICIV(
                root_dir=args.root_dir,
                sample_info_path=sample_info_path,
                lazy_mode=True,
                shuffle=False,
                table_mode="table_only",
                use_table_length_cache=False,
            )
            dataset.sample_info.extend(extra_dataset.sample_info)
        return dataset

    if args.dataset == "eicu":
        dataset = EICUDataset(
            root_dir=args.root_dir,
            processed_dir=args.processed_dir,
            sample_info_path=args.sample_info_path[0],
            task_name=args.task_name,
            lazy_mode=True,
            shuffle=False,
            table_mode="table_only",
        )
        for sample_info_path in args.sample_info_path[1:]:
            extra_dataset = EICUDataset(
                root_dir=args.root_dir,
                processed_dir=args.processed_dir,
                sample_info_path=sample_info_path,
                task_name=args.task_name,
                lazy_mode=True,
                shuffle=False,
                table_mode="table_only",
            )
            dataset.sample_info.extend(extra_dataset.sample_info)
        return dataset

    if args.dataset == "ehrshot":
        dataset = EHRSHOTDataset(
            root_dir=args.root_dir,
            sample_info_path=args.sample_info_path[0],
            task_name=args.task_name,
            lazy_mode=True,
            table_mode="table_only",
        )
        for sample_info_path in args.sample_info_path[1:]:
            extra_dataset = EHRSHOTDataset(
                root_dir=args.root_dir,
                sample_info_path=sample_info_path,
                task_name=args.task_name,
                lazy_mode=True,
                table_mode="table_only",
            )
            dataset.sample_info.extend(extra_dataset.sample_info)
        return dataset

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def get_markdown_text(dataset_name, dataset, idx):
    if dataset_name == "mimic_iv":
        return dataset[idx]["input"]
    sample_info = dataset.sample_info[idx]
    return dataset.free_text_input_process(sample_info)


def get_pooling_indices(model_path, tokenizer, input_ids, attention_mask):
    model_name_lower = model_path.lower()
    eos_id = None
    if "qwen" in model_name_lower or "ehr-r1" in model_name_lower:
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_id is not None:
        eos_mask = input_ids == eos_id
        if eos_mask.sum().item() > 0:
            seq_len = input_ids.size(1)
            indices = torch.arange(seq_len, device=input_ids.device)
            return (eos_mask * indices).argmax(dim=1)
    return attention_mask.sum(dim=1) - 1


def load_processed_keys(save_dir):
    processed_keys = set()
    for part_file in sorted(glob.glob(os.path.join(save_dir, "rank_[0-9]*_part_*.pt"))):
        processed_keys.update(torch.load(part_file, map_location="cpu", weights_only=True).keys())
    return processed_keys


def save_part(save_dir, rank, part_counter, embeddings):
    part_path = os.path.join(save_dir, f"rank_{rank}_part_{part_counter}.pt")
    torch.save(embeddings, part_path)
    print(f"[Rank {rank}] Saved part {part_counter}: {len(embeddings)} samples")


def merge_parts(save_dir):
    merged = {}
    part_files = sorted(glob.glob(os.path.join(save_dir, "rank_[0-9]*_part_*.pt")))
    for part_file in tqdm(part_files, desc="Merging"):
        merged.update(torch.load(part_file, map_location="cpu", weights_only=True))
    output_path = os.path.join(save_dir, "embeddings.pt")
    torch.save(merged, output_path)
    print(f"Saved merged embeddings to {output_path}")
    print(f"Total samples: {len(merged)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mimic_iv", "eicu", "ehrshot"], required=True)
    parser.add_argument("--model_path", type=str, default="/data/model_weights_public/BlueZeros/EHR-R1-1.7B")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--sample_info_path", type=str, nargs="+", required=True)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=32000)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--auto_num_processes", type=int, default=0)
    parser.add_argument("--disable_auto_launch", action="store_true")
    parser.add_argument("--merge_wait_seconds", type=int, default=7200)
    parser.add_argument("--merge_poll_seconds", type=float, default=2.0)
    args = parser.parse_args()

    maybe_auto_launch_distributed(args)
    accelerator = Accelerator()

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    if accelerator.is_main_process:
        print(f"Saving embeddings to: {save_dir}")
        print(f"Dataset: {args.dataset}")
        print(f"Distributed processes: {accelerator.num_processes}")
        for marker in glob.glob(os.path.join(save_dir, "rank_*.done")):
            os.remove(marker)
    accelerator.wait_for_everyone()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": accelerator.device},
    ).eval()

    dataset = build_dataset(args)
    processed_keys = load_processed_keys(save_dir)
    all_indices = [
        idx
        for idx, sample_info in enumerate(dataset.sample_info)
        if build_sample_key(args.dataset, sample_info) not in processed_keys
    ]
    my_indices = all_indices[accelerator.process_index::accelerator.num_processes]

    if accelerator.is_main_process:
        print(f"Total samples: {len(dataset.sample_info)}")
        print(f"Already processed samples: {len(processed_keys)}")
        print(f"Remaining samples: {len(all_indices)}")

    local_embeddings = {}
    batch_texts = []
    batch_keys = []
    part_counter = len(glob.glob(os.path.join(save_dir, f"rank_{accelerator.process_index}_part_*.pt")))
    batches_since_save = 0

    for idx in tqdm(my_indices, disable=not accelerator.is_main_process, desc="Generating"):
        sample_key = build_sample_key(args.dataset, dataset.sample_info[idx])
        text = get_markdown_text(args.dataset, dataset, idx)
        if tokenizer.chat_template:
            messages = [{"role": "user", "content": text}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        batch_texts.append(text)
        batch_keys.append(sample_key)

        if len(batch_texts) >= args.batch_size or idx == my_indices[-1]:
            inputs = tokenizer(
                batch_texts,
                max_length=args.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]

            pooling_indices = get_pooling_indices(
                args.model_path,
                tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            pooled = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), pooling_indices]

            for row_idx, key in enumerate(batch_keys):
                local_embeddings[key] = pooled[row_idx].cpu().to(torch.bfloat16)

            batch_texts = []
            batch_keys = []
            batches_since_save += 1

            if batches_since_save >= args.save_steps:
                save_part(save_dir, accelerator.process_index, part_counter, local_embeddings)
                local_embeddings.clear()
                batches_since_save = 0
                part_counter += 1

    if local_embeddings:
        save_part(save_dir, accelerator.process_index, part_counter, local_embeddings)

    done_marker = os.path.join(save_dir, f"rank_{accelerator.process_index}.done")
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write(str(time.time()))
    safe_destroy_process_group()

    if accelerator.is_main_process:
        deadline = time.time() + max(5, args.merge_wait_seconds)
        while time.time() < deadline:
            marker_count = len(glob.glob(os.path.join(save_dir, "rank_*.done")))
            if marker_count >= accelerator.num_processes:
                break
            time.sleep(max(0.1, args.merge_poll_seconds))
        merge_parts(save_dir)


if __name__ == "__main__":
    main()
