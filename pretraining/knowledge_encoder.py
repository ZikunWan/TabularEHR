from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.TableEncoder.text_encoder import KnowledgeGraphEncoderForTrainer


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank0_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


@dataclass
class Args:
    output_dir: str = "/data/zikun_workspace/checkpoints/pretraining/text_encoder"
    model_name_or_path: str = "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT"
    max_length: int = 256
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 42
    num_workers: int = 8
    freeze_bert: bool = False
    bf16: bool = True
    logging_steps: int = 50
    save_steps: int = 1000
    save_total_limit: int = 1
    report_to: str = "wandb"
    wandb_project: str = ""
    wandb_run_name: str = ""
    deepspeed: str = ""
    cache_only: bool = False

    concept_path: str = "/data/zikun_workspace/knowledge/CONCEPT.csv"
    concept_relationship_path: str = "/data/zikun_workspace/knowledge/CONCEPT_RELATIONSHIP.csv"
    triple_cache: str = "/data/zikun_workspace/.cache/pretraining/triples_cache"
    kg_max_triples: Optional[int] = None
    kg_num_negatives: int = 4
    kg_margin: float = 1.0
    kg_distance_p: int = 2
    kg_relation_reg: float = 1e-4


class ConceptRelationshipDataset(Dataset):
    def __init__(
        self,
        triple_cache: str,
        concepts: Dict[str, str],
        num_negatives: int,
        seed: int,
    ):
        self.concepts = concepts
        self.concept_ids = np.fromiter(
            (int(concept_id) for concept_id in concepts.keys()), dtype=np.int64
        )
        self.num_negatives = num_negatives
        self.seed = seed
        metadata_path = os.path.join(triple_cache, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.num_triples = int(metadata["num_triples"])
        self.relation2id: Dict[str, int] = metadata["relation2id"]
        self.head_ids = np.memmap(
            os.path.join(triple_cache, "head_ids.int64.bin"),
            dtype=np.int64,
            mode="r",
            shape=(self.num_triples,),
        )
        self.tail_ids = np.memmap(
            os.path.join(triple_cache, "tail_ids.int64.bin"),
            dtype=np.int64,
            mode="r",
            shape=(self.num_triples,),
        )
        self.relation_ids = np.memmap(
            os.path.join(triple_cache, "relation_ids.int64.bin"),
            dtype=np.int64,
            mode="r",
            shape=(self.num_triples,),
        )

    def __len__(self) -> int:
        return self.num_triples

    def _sample_negative(
        self, rng: random.Random, head_id: int, tail_id: int
    ) -> Tuple[str, bool]:
        # Build a negative triple by corrupting either the head or the tail concept.
        corrupt_head = rng.random() < 0.5
        original_id = head_id if corrupt_head else tail_id
        for _ in range(50):
            negative_id = int(self.concept_ids[rng.randrange(len(self.concept_ids))])
            if negative_id != original_id:
                return str(negative_id), corrupt_head
        return str(int(self.concept_ids[rng.randrange(len(self.concept_ids))])), corrupt_head

    def __getitem__(self, idx: int) -> Dict[str, object]:
        head_id = int(self.head_ids[idx])
        tail_id = int(self.tail_ids[idx])
        relation_id = int(self.relation_ids[idx])
        rng = random.Random(self.seed + idx)
        negative_names = []
        negative_is_head = []
        for _ in range(self.num_negatives):
            negative_id, is_head = self._sample_negative(rng, head_id, tail_id)
            negative_names.append(self.concepts[negative_id])
            negative_is_head.append(is_head)

        return {
            "head_name": self.concepts[str(head_id)],
            "tail_name": self.concepts[str(tail_id)],
            "relation_id": relation_id,
            "negative_names": negative_names,
            "negative_is_head": negative_is_head,
        }


def count_file_chunks(path: str, chunksize: int) -> int:
    lines = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            lines += block.count(b"\n")
    data_lines = max(0, lines - 1)
    return max(1, (data_lines + chunksize - 1) // chunksize)

def load_concept_names(concept_path: str) -> Dict[str, str]:
    concepts: Dict[str, str] = {}
    total_chunks = count_file_chunks(concept_path, 1_000_000)
    reader = pd.read_csv(
        concept_path,
        sep="\t",
        dtype=str,
        usecols=["concept_id", "concept_name"],
        chunksize=1_000_000,
        keep_default_na=False,
    )
    for chunk in tqdm(
        reader,
        desc="Reading CONCEPT",
        total=total_chunks,
        unit="chunk",
        disable=not is_main_process(),
    ):
        for concept_id, concept_name in chunk.itertuples(index=False, name=None):
            concept_id = str(concept_id).strip()
            concept_name = str(concept_name).strip()
            if concept_id and concept_name:
                concepts[concept_id] = concept_name
    return concepts


def build_triples(
    args: Args, triple_cache: str, concepts: Dict[str, str]
) -> int:
    os.makedirs(triple_cache, exist_ok=True)
    head_path = os.path.join(triple_cache, "head_ids.int64.bin")
    tail_path = os.path.join(triple_cache, "tail_ids.int64.bin")
    relation_path = os.path.join(triple_cache, "relation_ids.int64.bin")
    metadata_path = os.path.join(triple_cache, "metadata.json")
    concept_ids = set(concepts)
    relation2id: Dict[str, int] = {}
    written = 0
    total_chunks = count_file_chunks(args.concept_relationship_path, 1_000_000)

    reader = pd.read_csv(
        args.concept_relationship_path,
        sep="\t",
        dtype=str,
        usecols=[
            "concept_id_1",
            "concept_id_2",
            "relationship_id",
        ],
        chunksize=1_000_000,
        keep_default_na=False,
    )
    with open(head_path, "wb") as head_out, open(tail_path, "wb") as tail_out, open(
        relation_path, "wb"
    ) as relation_out:
        for chunk in tqdm(
            reader,
            desc="Building triples",
            total=total_chunks,
            unit="chunk",
            disable=not is_main_process(),
        ):
            chunk = chunk[
                chunk["concept_id_1"].ne("")
                & chunk["concept_id_2"].ne("")
                & chunk["relationship_id"].ne("")
                & chunk["concept_id_1"].ne(chunk["concept_id_2"])
                & chunk["concept_id_1"].isin(concept_ids)
                & chunk["concept_id_2"].isin(concept_ids)
            ]
            if args.kg_max_triples is not None:
                remaining = args.kg_max_triples - written
                if remaining <= 0:
                    break
                chunk = chunk.head(remaining)

            for relationship_id in chunk["relationship_id"].unique():
                if relationship_id not in relation2id:
                    relation2id[relationship_id] = len(relation2id)

            head_ids = chunk["concept_id_1"].astype(np.int64).to_numpy()
            tail_ids = chunk["concept_id_2"].astype(np.int64).to_numpy()
            relation_ids = (
                chunk["relationship_id"].map(relation2id).astype(np.int64).to_numpy()
            )
            head_ids.tofile(head_out)
            tail_ids.tofile(tail_out)
            relation_ids.tofile(relation_out)
            written += len(chunk)
            if args.kg_max_triples is not None and written >= args.kg_max_triples:
                break

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_triples": written,
                "relation2id": relation2id,
                "format": "int64_memmap_v1",
                "files": {
                    "head_ids": os.path.basename(head_path),
                    "tail_ids": os.path.basename(tail_path),
                    "relation_ids": os.path.basename(relation_path),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    rank0_print(f"Triple cache: {triple_cache} ({written} triples)")
    rank0_print(f"Relations: {len(relation2id)}")
    return written


def make_kg_collate_fn(tokenizer, max_length: int):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        head_tokens = tokenizer(
            [str(item["head_name"]) for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tail_tokens = tokenizer(
            [str(item["tail_name"]) for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        num_negatives = len(batch[0]["negative_names"])
        flat_negative_names: List[str] = []
        negative_is_head: List[List[bool]] = []
        for item in batch:
            flat_negative_names.extend(str(name) for name in item["negative_names"])
            negative_is_head.append(list(item["negative_is_head"]))
        negative_tokens = tokenizer(
            flat_negative_names,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_size, negative_seq_len = len(batch), negative_tokens["input_ids"].size(1)

        output = {
            "head_input_ids": head_tokens["input_ids"],
            "head_attention_mask": head_tokens["attention_mask"],
            "tail_input_ids": tail_tokens["input_ids"],
            "tail_attention_mask": tail_tokens["attention_mask"],
            "relation_ids": torch.tensor(
                [int(item["relation_id"]) for item in batch], dtype=torch.long
            ),
            "negative_input_ids": negative_tokens["input_ids"].view(
                batch_size, num_negatives, negative_seq_len
            ),
            "negative_attention_mask": negative_tokens["attention_mask"].view(
                batch_size, num_negatives, negative_seq_len
            ),
            "negative_is_head": torch.tensor(negative_is_head, dtype=torch.bool),
        }
        if "token_type_ids" in head_tokens:
            output["head_token_type_ids"] = head_tokens["token_type_ids"]
        if "token_type_ids" in tail_tokens:
            output["tail_token_type_ids"] = tail_tokens["token_type_ids"]
        if "token_type_ids" in negative_tokens:
            output["negative_token_type_ids"] = negative_tokens["token_type_ids"].view(
                batch_size, num_negatives, negative_seq_len
            )
        return output

    return collate


def configure_wandb(args: Args) -> None:
    if args.report_to != "wandb":
        return
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

def run_training(args: Args) -> None:
    random.seed(args.seed)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.kg_num_negatives <= 0:
        raise ValueError("kg_num_negatives must be greater than 0.")

    concepts = load_concept_names(args.concept_path)
    triple_cache = args.triple_cache
    if not os.path.exists(os.path.join(triple_cache, "metadata.json")):
        build_triples(args, triple_cache, concepts)
    if args.cache_only:
        rank0_print(f"Triple cache is ready: {triple_cache}")
        return

    dataset = ConceptRelationshipDataset(
        triple_cache=triple_cache,
        concepts=concepts,
        num_negatives=args.kg_num_negatives,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after filtering concept relationships.")

    train_dataset = dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    collate_fn = make_kg_collate_fn(tokenizer, args.max_length)
    model = KnowledgeGraphEncoderForTrainer(
        args.model_name_or_path,
        num_relations=len(dataset.relation2id),
        margin=args.kg_margin,
        distance_p=args.kg_distance_p,
        relation_reg=args.kg_relation_reg,
        freeze_bert=args.freeze_bert,
    )
    configure_wandb(args)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="no",
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        prediction_loss_only=True,
        bf16=args.bf16,
        report_to=[] if args.report_to == "none" else [args.report_to],
        run_name=args.wandb_run_name or None,
        deepspeed=args.deepspeed or None,
        log_on_each_node=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=tokenizer,
    )
    rank0_print(
        f"Knowledge graph triples: train={len(train_dataset)}, "
        f"relations={len(dataset.relation2id)}"
    )
    rank0_print(f"Concept text: concept_name")
    rank0_print(f"Model: {args.model_name_or_path}")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(
        {
            "state_dict": trainer.model.state_dict(),
            "model_name_or_path": args.model_name_or_path,
            "args": asdict(args),
            "relation2id": dataset.relation2id,
        },
        os.path.join(args.output_dir, "best.pt"),
    )
    rank0_print(f"Saved checkpoint: {os.path.join(args.output_dir, 'best.pt')}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Knowledge graph encoder pretraining")
    for field_name, field_def in Args.__dataclass_fields__.items():
        default = field_def.default
        arg_type = type(default)
        if arg_type is bool:
            parser.add_argument(f"--{field_name}", action="store_true", default=default)
        elif field_name == "kg_max_triples":
            parser.add_argument(f"--{field_name}", type=str, default=default)
        else:
            parser.add_argument(f"--{field_name}", type=arg_type, default=default)
    parser.add_argument("--local_rank", type=int, default=-1)
    values = vars(parser.parse_args())
    values.pop("local_rank", None)
    if isinstance(values["kg_max_triples"], str):
        raw_value = values["kg_max_triples"].strip()
        values["kg_max_triples"] = (
            None if raw_value.lower() == "none" else int(raw_value)
        )
    return Args(**values)


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
