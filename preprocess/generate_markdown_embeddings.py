"""
Build patient-state contrastive caches.

Outputs in --output_dir:
    - embeddings.pt: {state_key: pooled full-state markdown embedding}
    - state_tables.pt: {state_key: complementary table views for the same patient state}
    - metadata.json: cache configuration for validation during training
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dataset.ehrshot.ehrshot_dataset import EHRSHOTDataset
from dataset.eicu.eicu_dataset import EICUDataset
from dataset.mimic.mimic_dataset import MIMICIV, read_parquet


def build_state_key(dataset_name: str, sample_info: dict[str, Any]) -> str:
    if sample_info.get("sample_id") is not None:
        return str(sample_info["sample_id"])
    if dataset_name == "mimic_iv":
        return (
            f"mimic_iv|{sample_info.get('subject_id', '')}|"
            f"{sample_info.get('task', '')}|"
            f"{sample_info.get('context_begin', '')}|"
            f"{sample_info.get('context_end', '')}"
        )
    if dataset_name == "eicu":
        return (
            f"eicu|{sample_info.get('patient_id', '')}|"
            f"{sample_info.get('icustay_id', '')}|"
            f"{sample_info.get('task_name', '')}|"
            f"{sample_info.get('obs_hours', '')}|"
            f"{sample_info.get('gap_hours', '')}|"
            f"{sample_info.get('pred_hours', '')}"
        )
    if dataset_name == "ehrshot":
        return (
            f"ehrshot|{sample_info.get('patient_id', '')}|"
            f"{sample_info.get('task_name', '')}|"
            f"{sample_info.get('prediction_time', '')}"
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_subject_id(dataset_name: str, sample_info: dict[str, Any]) -> str:
    if dataset_name == "mimic_iv":
        return str(sample_info.get("subject_id", ""))
    if dataset_name in {"eicu", "ehrshot"}:
        return str(sample_info.get("patient_id", ""))
    raise ValueError(f"Unsupported dataset: {dataset_name}")


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
    print(f"Auto-launching distributed cache build with {num_procs} GPUs: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    sys.exit(0)


def safe_destroy_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


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


def expected_metadata(args) -> dict[str, Any]:
    return {
        "cache_schema_version": 4,
        "dataset": args.dataset,
        "sample_info_path": list(args.sample_info_path),
        "task_name": args.task_name,
        "model_path": args.model_path,
        "max_length": int(args.max_length),
        "max_table_len": None if args.max_table_len is None else int(args.max_table_len),
        "min_table_rows": int(args.min_table_rows),
        "view_strategy": "patient_state_category_views",
        "markdown_text": "full_patient_state",
    }


def load_existing_metadata(save_dir: str):
    metadata_path = os.path.join(save_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_cache_artifacts(save_dir: str):
    patterns = [
        "rank_[0-9]*_part_*.pt",
        "rank_*.done",
        "embeddings.pt",
        "state_tables.pt",
        "view_specs.pt",
        "metadata.json",
    ]
    for pattern in patterns:
        for path in glob.glob(os.path.join(save_dir, pattern)):
            try:
                os.remove(path)
            except OSError:
                pass


def load_processed_keys(save_dir: str):
    processed_keys = set()
    for part_file in sorted(glob.glob(os.path.join(save_dir, "rank_[0-9]*_part_*.pt"))):
        processed_keys.update(torch.load(part_file, map_location="cpu", weights_only=False).keys())
    return processed_keys


def save_part(save_dir: str, rank: int, part_counter: int, records: dict[str, Any]):
    part_path = os.path.join(save_dir, f"rank_{rank}_part_{part_counter}.pt")
    torch.save(records, part_path)
    print(f"[Rank {rank}] Saved part {part_counter}: {len(records)} samples")


def merge_parts(save_dir: str, metadata: dict[str, Any]):
    merged_embeddings = {}
    merged_state_tables = {}
    part_files = sorted(glob.glob(os.path.join(save_dir, "rank_[0-9]*_part_*.pt")))
    for part_file in tqdm(part_files, desc="Merging"):
        records = torch.load(part_file, map_location="cpu", weights_only=False)
        for key, record in records.items():
            merged_embeddings[key] = record["embedding"]
            merged_state_tables[key] = record["state_table"]

    torch.save(merged_embeddings, os.path.join(save_dir, "embeddings.pt"))
    torch.save(merged_state_tables, os.path.join(save_dir, "state_tables.pt"))
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, indent=2, sort_keys=True)
    print(f"Saved markdown embeddings to {os.path.join(save_dir, 'embeddings.pt')}")
    print(f"Saved patient-state tables to {os.path.join(save_dir, 'state_tables.pt')}")
    print(f"Total cached samples: {len(merged_embeddings)}")


def serialize_table_records(measurement_table: pd.DataFrame) -> list[dict[str, str]]:
    if measurement_table is None or measurement_table.empty:
        return []

    table = measurement_table.copy()
    if "Time" in table.columns:
        table["Time"] = pd.to_datetime(table["Time"], errors="coerce")
        table["Time"] = table["Time"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    records = []
    for record in table.to_dict(orient="records"):
        serialized = {}
        for key, value in record.items():
            serialized[str(key)] = "" if pd.isna(value) else str(value)
        records.append(serialized)
    return records


def normalize_table(table: pd.DataFrame, max_table_len: int | None):
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


def split_patient_state_views(table: pd.DataFrame, min_table_rows: int):
    table = table.reset_index(drop=True)
    category = table["Category"].fillna("").astype(str).str.lower()
    demo = table[category.str.contains("person|demographic", regex=True)]
    non_demo = table[~category.str.contains("person|demographic", regex=True)]
    if len(non_demo) < 2:
        return None

    anchor_mask = non_demo["Category"].fillna("").astype(str).str.lower().str.contains(
        "measurement|observation|lab|vital|chart",
        regex=True,
    )
    anchor_core = non_demo[anchor_mask]
    positive_core = non_demo[~anchor_mask]

    if anchor_core.empty or positive_core.empty:
        anchor_core = non_demo.iloc[::2]
        positive_core = non_demo.iloc[1::2]

    anchor = pd.concat([demo, anchor_core], ignore_index=True)
    positive = pd.concat([demo, positive_core], ignore_index=True)
    if len(anchor) < min_table_rows or len(positive) < min_table_rows:
        return None

    anchor = anchor.sort_values("Time").reset_index(drop=True)
    positive = positive.sort_values("Time").reset_index(drop=True)
    return anchor, positive


def flush_batch(model, tokenizer, args, batch_texts, batch_keys, batch_state_tables, local_records):
    if len(batch_texts) == 0:
        return

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
        local_records[key] = {
            "embedding": pooled[row_idx].cpu().to(torch.bfloat16),
            "state_table": batch_state_tables[row_idx],
        }


def build_mimic_state(dataset: MIMICIV, sample_info: dict[str, Any], args):
    subject_id = str(sample_info["subject_id"])
    patient_trajectory_list = read_parquet(f"{dataset.ehr_dir}/{subject_id}.parquet")
    context_begin = int(sample_info["context_begin"])
    context_end = int(sample_info["context_end"])
    task_name = sample_info["task"]

    trajectory_events = [
        item
        for item in patient_trajectory_list[context_begin:context_end]
        if item["file_name"] not in dataset.task_schema[task_name]["bid_event"]
        and item["file_name"] not in {"admissions", "patients"}
    ]
    structured_events = [
        item for item in trajectory_events if item.get("file_name") not in {"discharge", "radiology"}
    ]
    if len(structured_events) < 2:
        return None

    table = dataset.structed_EHR_input_process(structured_events, patient_trajectory_list)
    table = normalize_table(table, args.max_table_len)
    if table is None or len(table) < args.min_table_rows:
        return None

    split_views = split_patient_state_views(table, args.min_table_rows)
    if split_views is None:
        return None
    anchor_table, positive_table = split_views

    patient_text = ""
    if patient_trajectory_list and patient_trajectory_list[0].get("file_name") == "patients":
        patient_text = dataset.convertor.input_process(patient_trajectory_list[0])
    prefix_text_list = [patient_text] if isinstance(patient_text, str) and patient_text.strip() else []
    markdown_text = dataset.free_text_input_process(structured_events, prefix_text_list)
    if not isinstance(markdown_text, str) or not markdown_text.strip():
        return None

    return markdown_text, anchor_table, positive_table


def build_generic_state(dataset, sample_info: dict[str, Any], idx: int, args):
    sample = dataset[idx]
    table = normalize_table(sample.get("measurement_table"), args.max_table_len)
    if table is None or len(table) < args.min_table_rows:
        return None

    split_views = split_patient_state_views(table, args.min_table_rows)
    if split_views is None:
        return None
    anchor_table, positive_table = split_views

    markdown_text = dataset.free_text_input_process(sample_info)
    if not isinstance(markdown_text, str) or not markdown_text.strip():
        return None
    return markdown_text, anchor_table, positive_table


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
    parser.add_argument("--max_table_len", type=int, default=4096)
    parser.add_argument("--min_table_rows", type=int, default=2)
    parser.add_argument("--auto_num_processes", type=int, default=0)
    parser.add_argument("--disable_auto_launch", action="store_true")
    parser.add_argument("--merge_wait_seconds", type=int, default=7200)
    parser.add_argument("--merge_poll_seconds", type=float, default=2.0)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    maybe_auto_launch_distributed(args)
    accelerator = Accelerator()

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    metadata = expected_metadata(args)
    existing_metadata = load_existing_metadata(save_dir)
    final_embeddings = os.path.join(save_dir, "embeddings.pt")
    final_state_tables = os.path.join(save_dir, "state_tables.pt")

    if (
        not args.force_rebuild
        and existing_metadata == metadata
        and os.path.exists(final_embeddings)
        and os.path.exists(final_state_tables)
    ):
        if accelerator.is_main_process:
            print(f"Reusing existing patient-state cache in {save_dir}")
        return

    if accelerator.is_main_process:
        if args.force_rebuild or (existing_metadata is not None and existing_metadata != metadata):
            print("Clearing stale cache artifacts before rebuild...")
            clear_cache_artifacts(save_dir)
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
        if build_state_key(args.dataset, sample_info) not in processed_keys
    ]
    my_indices = all_indices[accelerator.process_index::accelerator.num_processes]

    if accelerator.is_main_process:
        print(f"Saving patient-state cache to: {save_dir}")
        print(f"Dataset: {args.dataset}")
        print(f"Distributed processes: {accelerator.num_processes}")
        print(f"Total samples: {len(dataset.sample_info)}")
        print(f"Already cached: {len(processed_keys)}")
        print(f"Remaining samples: {len(all_indices)}")

    local_records = {}
    batch_texts = []
    batch_keys = []
    batch_state_tables = []
    part_counter = len(glob.glob(os.path.join(save_dir, f"rank_{accelerator.process_index}_part_*.pt")))
    batches_since_save = 0

    iterator = tqdm(my_indices, disable=not accelerator.is_main_process, desc="Building patient states")
    for idx in iterator:
        sample_info = dataset.sample_info[idx]
        sample_key = build_state_key(args.dataset, sample_info)
        if args.dataset == "mimic_iv":
            state = build_mimic_state(dataset, sample_info, args)
        else:
            state = build_generic_state(dataset, sample_info, idx, args)
        if state is None:
            continue

        markdown_text, anchor_table, positive_table = state
        if tokenizer.chat_template:
            markdown_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": markdown_text}],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

        batch_texts.append(markdown_text)
        batch_keys.append(sample_key)
        batch_state_tables.append(
            {
                "subject_id": build_subject_id(args.dataset, sample_info),
                "anchor_table_records": serialize_table_records(anchor_table),
                "positive_table_records": serialize_table_records(positive_table),
                "table_length": int(len(anchor_table) + len(positive_table)),
                "anchor_length": int(len(anchor_table)),
                "positive_length": int(len(positive_table)),
            }
        )

        if len(batch_texts) >= args.batch_size:
            flush_batch(model, tokenizer, args, batch_texts, batch_keys, batch_state_tables, local_records)
            batch_texts = []
            batch_keys = []
            batch_state_tables = []
            batches_since_save += 1

            if batches_since_save >= args.save_steps:
                save_part(save_dir, accelerator.process_index, part_counter, local_records)
                local_records.clear()
                batches_since_save = 0
                part_counter += 1

    flush_batch(model, tokenizer, args, batch_texts, batch_keys, batch_state_tables, local_records)
    if local_records:
        save_part(save_dir, accelerator.process_index, part_counter, local_records)

    done_marker = os.path.join(save_dir, f"rank_{accelerator.process_index}.done")
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write(str(time.time()))
    safe_destroy_process_group()

    if accelerator.is_main_process:
        deadline = time.time() + max(5, args.merge_wait_seconds)
        while time.time() < deadline:
            if len(glob.glob(os.path.join(save_dir, "rank_*.done"))) >= accelerator.num_processes:
                break
            time.sleep(max(0.1, args.merge_poll_seconds))
        merge_parts(save_dir, metadata)


if __name__ == "__main__":
    main()
