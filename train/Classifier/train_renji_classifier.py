import os
import sys
import json
import logging
import torch
import torch.nn as nn
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from peft import LoraConfig, get_peft_model, PeftModel
import glob
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from transformers.utils import logging as hf_logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

def rank0_print(*args, **kwargs):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in [-1, 0]:
        print(*args, **kwargs)


def quiet_non_main_process_logs():
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank not in [-1, 0]:
        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)
        logging.getLogger("deepspeed").setLevel(logging.ERROR)

from dataset.renji_dataset import RenjiDataset
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.query_classifier import TaskQueryClassificationModel
from utils.weight_loader import load_model_weights
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.collate import create_query_collate_fn
from utils.query_embedding import build_query_embeddings

@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to pre-trained model checkpoint (safetensors or bin) or TAPAS base"})
    
    # LoRA options
    use_lora: bool = field(default=False, metadata={"help": "Apply LoRA adapters to encoder linear layers via peft"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (scaling = alpha/r)"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="qkv,proj,w12,w3",
        metadata={"help": "Comma-separated list of Linear layer names to apply LoRA to."}
    )


@dataclass
class DataArguments:
    max_table_len: int = field(metadata={"help": "Keep only the most recent N table rows before encoding"})
    data_dir: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/Renji")
    embedding_cache: str = field(default="/home/ma-user/sfs_turbo/sai6/zkwan/.cache/embeddings/renji/text_embeddings.pt")
    max_train_samples: Optional[int] = field(default=None)
    type_vocab_file: str = field(default="data/type_vocab.json")
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_embeddings.pt")
    query_text_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/text_encoder_stage2/epoch_5.pt")
    query_text_encoder_base_model: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=128)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="Renji")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    quiet_non_main_process_logs()
    
    training_args.lr_scheduler_type = "cosine"
    if training_args.warmup_steps == 0:
        training_args.warmup_steps = 100
    training_args.warmup_ratio = 0.0
    training_args.weight_decay = 0.01
    training_args.seed = 42
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 10
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.save_total_limit = 2
    training_args.bf16 = True
    training_args.dataloader_num_workers = 4
    training_args.remove_unused_columns = False
    training_args.report_to = ["wandb"]
    training_args.save_safetensors = True
    
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    training_args.ddp_find_unused_parameters = False
    set_seed(training_args.seed)

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]
    
    vocab_path = os.path.join(project_root, data_args.type_vocab_file)
    with open(vocab_path, 'r') as f:
        type_vocab = json.load(f)

    train_dataset = RenjiDataset(
        root_dir=data_args.data_dir, split="train", table_mode="table_only", shuffle=True,
        max_samples=data_args.max_train_samples, task_mode='multi_label'
    )

    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        num_points=len(RenjiDataset.ALL_POINTS),
        num_metrics=len(RenjiDataset.ALL_METRICS),
        num_classes=len(RenjiDataset.ALL_POINTS) * len(RenjiDataset.ALL_METRICS),
        problem_type="multi_label_classification"
    )
    
    query_texts = {}
    query_template = RenjiDataset.TASK_INFO["multi_label_prediction"]["instruction_template"]
    for point_key in RenjiDataset.ALL_POINTS:
        _, _, readable_point = RenjiDataset.TASK_PREDICTION_POINTS[point_key]
        instruction = query_template.format(prediction_point=f"{readable_point} post-transplant")
        query_texts[instruction] = instruction
    query_embeddings_by_text, query_dim = build_query_embeddings(
        query_texts,
        data_args.query_embedding_cache,
        data_args.query_text_encoder_path,
        data_args.query_text_encoder_base_model,
        data_args.query_max_length,
    )

    model = TaskQueryClassificationModel(
        config=encoder_config,
        embedding_matrix=embedding_matrix,
        query_dim=query_dim,
    )
    model = load_model_weights(model, model_args.pretrained_path, use_lora=False, is_trainable=False)
    
    if model_args.use_lora:
        existing_ckpts = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
        if training_args.resume_from_checkpoint and existing_ckpts:
            latest_ckpt = existing_ckpts[-1]
            rank0_print(f"Loading LoRA adapter weights from checkpoint: {latest_ckpt}")
            model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=True)
        else:
            target_modules = [m.strip() for m in model_args.lora_target_modules.split(',')]
            lora_cfg = LoraConfig(
                r=model_args.lora_r, lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, bias="none",
                target_modules=target_modules,
                modules_to_save=["classifier", "query_head"],
            )
            model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset,
        data_collator=create_query_collate_fn(
            type_vocab,
            max_table_len=data_args.max_table_len,
            text_to_idx=text_to_idx,
            pad_idx=pad_idx,
            query_embeddings_by_text=query_embeddings_by_text,
        ),
    )

    resume_ckpt = None
    if training_args.resume_from_checkpoint:
        rfc = training_args.resume_from_checkpoint
        if isinstance(rfc, str) and rfc.lower() not in ("true", "1", "yes"): resume_ckpt = rfc
        else:
            ckpt_dirs = sorted(glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[-1]))
            if ckpt_dirs: resume_ckpt = ckpt_dirs[-1]

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model()


if __name__ == "__main__":
    main()
