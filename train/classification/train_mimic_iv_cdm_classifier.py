import os
import sys
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
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
from models.query_classifier import TaskQueryClassificationModel
from utils.load_embedding import (
    build_embedding_matrix,
    build_text_to_idx,
    build_vocab_keys,
    get_special_token_indices,
    load_embedding_cache,
)
from utils.weight_loader import apply_fine_tune_mode, load_encoder_weights
from utils.collate import create_query_collate_fn
from utils.load_embedding import build_task_query_embeddings

LABEL_MAP = {
    'appendicitis': 0,
    'cholecystitis': 1,
    'diverticulitis': 2,
    'pancreatitis': 3,
}
NUM_CLASSES = len(LABEL_MAP)


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "Path to pre-trained model checkpoint"})
    fine_tune_mode: str = field(
        default="full_fine_tune",
        metadata={"help": "Fine-tuning mode: full_fine_tune or linear_probe"},
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
        default="/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt",
        metadata={"help": "Path to pre-computed embedding cache"}
    )
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Limit training samples"})
    type_vocab_file: str = field(
        default="data/type_vocab.json",
        metadata={"help": "Path to type vocabulary JSON file"}
    )
    query_embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt")
    query_encoder: str = field(default="llm")
    query_llm_model_path: str = field(default="/home/ma-user/modelarts/user-job-dir/LiverTransplantation/model_weights/BlueZeros/EHR-R1-1.7B")
    knowledge_encoder_path: str = field(default="/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt")
    knowledge_encoder_base_model_path: str = field(default="/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT")
    query_max_length: int = field(default=512)
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
        lazy_mode=False,
        shuffle=False,
        max_samples=data_args.max_train_samples,
    )
    rank0_print(f"Train dataset size: {len(train_dataset)}")

    task_info = get_task_info()[data_args.task_name]
    query_key = f"mimic_iv_cdm:{data_args.task_name}"
    query_embeddings, query_dim = build_task_query_embeddings(
        query_texts={query_key: task_info["instruction"]},
        cache_path=data_args.query_embedding_cache,
        query_encoder=data_args.query_encoder,
        max_length=data_args.query_max_length,
        query_llm_model_path=data_args.query_llm_model_path,
        knowledge_encoder_path=data_args.knowledge_encoder_path,
        knowledge_encoder_base_model_path=data_args.knowledge_encoder_base_model_path,
    )
    rank0_print(f"Query encoder={data_args.query_encoder}, query_dim={query_dim}")

    # 4. Model
    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=query_dim,
        num_classes=NUM_CLASSES,
        problem_type="single_label_classification"
    )

    model = TaskQueryClassificationModel(
        config=encoder_config,
        embedding_matrix=embedding_matrix,
        query_dim=query_dim,
    )

    # 5. Load pre-trained weights (optional)
    if model_args.pretrained_path:
        model = load_encoder_weights(model, model_args.pretrained_path)
    model = apply_fine_tune_mode(model, model_args.fine_tune_mode, log_fn=rank0_print)

    # 6. Trainer
    collate_fn = create_query_collate_fn(
        type_vocab,
        label_map=LABEL_MAP,
        max_table_len=data_args.max_table_len,
        text_to_idx=text_to_idx,
        pad_idx=pad_idx,
        query_embed=query_embeddings[query_key],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )

    # 8. Train
    rank0_print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    rank0_print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()


if __name__ == "__main__":
    main()
