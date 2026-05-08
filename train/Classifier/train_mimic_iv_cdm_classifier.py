import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from peft import LoraConfig, get_peft_model
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

from dataset.mimic_iv_cdm.mimic_iv_cdm_dataset import MIMICIVCDM
from dataset.mimic_iv_cdm.task_info import get_task_info
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.TableEncoder.query_classifier import TaskQueryClassificationModel
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.samplers import TrainerWithBatchSampler, build_train_batch_sampler
from utils.weight_loader import load_model_weights
from utils.collate import create_query_collate_fn
from utils.query_embedding import build_query_embeddings

LABEL_MAP = {
    'appendicitis': 0,
    'cholecystitis': 1,
    'diverticulitis': 2,
    'pancreatitis': 3,
}
NUM_CLASSES = len(LABEL_MAP)


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to pre-trained model checkpoint (safetensors or bin)"})
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
    data_dir: str = field(
        default="/data/EHR_data_public/mimic-iv-cdm",
        metadata={"help": "Root directory for MIMIC-IV-CDM data"}
    )
    task_name: str = field(
        default="MIMIC-IV-CDM Main Disease Diagnoses",
        metadata={"help": "Task name: 'MIMIC-IV-CDM Main Disease Diagnoses' or 'MIMIC-IV-CDM ICD Code Diagnoses'"}
    )
    embedding_cache: str = field(
        default="/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings.pt",
        metadata={"help": "Path to pre-computed embedding cache"}
    )
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Limit training samples"})
    type_vocab_file: str = field(
        default="/data/zikun_workspace/code/data/type_vocab.json",
        metadata={"help": "Path to type vocabulary JSON file"}
    )
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_embeddings.pt")
    query_text_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/text_encoder_stage2/epoch_5.pt")
    query_text_encoder_base_model: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=128)
    max_tokens_per_batch: Optional[int] = field(default=None, metadata={"help": "Enable ApproxBatchSampler when >0. This caps padded tokens per batch."})
    use_sortish_sampler: bool = field(default=True, metadata={"help": "Whether to use SortishSampler before ApproxBatchSampler packing."})
    sortish_chunk_factor: int = field(default=50, metadata={"help": "Sortish chunk factor. Larger means more sorting, less randomness."})

@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="MIMIC-IV-CDM", metadata={"help": "W&B project name."})

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    quiet_non_main_process_logs()

    # Default training settings
    training_args.lr_scheduler_type = "cosine"
    if training_args.warmup_steps == 0:
        training_args.warmup_steps = 100
    training_args.warmup_ratio = 0.0
    training_args.weight_decay = 0.01
    training_args.seed = 42
    training_args.eval_strategy = "no"
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 10
    training_args.save_strategy = "steps"
    training_args.save_total_limit = 2
    training_args.bf16 = True
    training_args.dataloader_num_workers = 8
    training_args.remove_unused_columns = False
    training_args.report_to = ["wandb"]
    training_args.save_safetensors = True

    set_seed(training_args.seed)

    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # 1. Load Embedding Cache
    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    vocab_keys = build_vocab_keys(embedding_cache)
    text_to_idx = build_text_to_idx(vocab_keys)
    embedding_matrix = build_embedding_matrix(embedding_cache, vocab_keys)
    pad_idx = get_special_token_indices(text_to_idx)["pad_idx"]

    # 2. Load Type Vocab
    type_vocab = None
    with open(data_args.type_vocab_file, 'r') as f:
        type_vocab = json.load(f)

    # 3. Load Dataset (train only; val+test are reserved for final evaluation)
    rank0_print(f"Loading MIMIC-IV-CDM dataset from {data_args.data_dir}...")
    train_dataset = MIMICIVCDM(
        root_dir=data_args.data_dir,
        split="train",
        task_name=data_args.task_name,
        table_mode="table_only",
        lazy_mode=False,
        shuffle=False,
        max_samples=data_args.max_train_samples,
    )
    rank0_print(f"Train dataset size: {len(train_dataset)}")

    # 4. Model
    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        num_classes=NUM_CLASSES,
        problem_type="single_label_classification"
    )

    task_info = get_task_info()[data_args.task_name]
    query_key = f"mimic_iv_cdm:{data_args.task_name}"
    query_embeddings, query_dim = build_query_embeddings(
        {query_key: task_info["instruction"]},
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

    # 5. Load pre-trained weights (optional)
    if model_args.pretrained_path:
        model = load_model_weights(model, model_args.pretrained_path, use_lora=False, is_trainable=False)

    # 6. LoRA (optional)
    if model_args.use_lora:
        target_modules = [m.strip() for m in model_args.lora_target_modules.split(',')]
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=["classifier", "query_head"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        rank0_print("LoRA applied.")

    # 7. Trainer
    collate_fn = create_query_collate_fn(
        type_vocab,
        label_map=LABEL_MAP,
        max_table_len=data_args.max_table_len,
        text_to_idx=text_to_idx,
        pad_idx=pad_idx,
        query_embed=query_embeddings[query_key],
    )

    trainer_cls = Trainer
    trainer_kwargs = {}
    if data_args.max_tokens_per_batch is not None and data_args.max_tokens_per_batch > 0:
        train_batch_sampler = build_train_batch_sampler(
            dataset=train_dataset,
            per_device_batch_size=training_args.per_device_train_batch_size,
            max_tokens_per_batch=data_args.max_tokens_per_batch,
            use_sortish_sampler=data_args.use_sortish_sampler,
            sortish_chunk_factor=data_args.sortish_chunk_factor,
            shuffle=True,
            seed=training_args.seed,
            drop_last=training_args.dataloader_drop_last,
        )
        trainer_cls = TrainerWithBatchSampler
        trainer_kwargs["train_batch_sampler"] = train_batch_sampler
        rank0_print(
            f"Enabled dynamic token batching: max_tokens_per_batch={data_args.max_tokens_per_batch}, "
            f"use_sortish_sampler={data_args.use_sortish_sampler}"
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        **trainer_kwargs,
    )

    # 8. Train
    rank0_print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    rank0_print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()


if __name__ == "__main__":
    main()
