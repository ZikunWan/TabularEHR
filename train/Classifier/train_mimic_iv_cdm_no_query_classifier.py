import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, get_peft_model
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.utils import logging as hf_logging

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
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.encoder_classifier import LongTableEncoderClassifier
from utils.collate import create_collate_fn
from utils.load_embedding import load_embedding_cache
from utils.samplers import TrainerWithBatchSampler, build_train_batch_sampler
from utils.weight_loader import load_model_weights


LABEL_MAP = {
    "appendicitis": 0,
    "cholecystitis": 1,
    "diverticulitis": 2,
    "pancreatitis": 3,
}
NUM_CLASSES = len(LABEL_MAP)


@dataclass
class ModelArguments:
    pretrained_path: Optional[str] = field(default=None)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="qkv,proj,w12,w3")
    dim_out: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    max_table_len: int = field(metadata={"help": "Keep only the most recent N table rows before encoding"})
    data_dir: str = field(default="/data/EHR_data_public/mimic-iv-cdm")
    task_name: str = field(default="MIMIC-IV-CDM Main Disease Diagnoses")
    embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt")
    max_train_samples: Optional[int] = field(default=None)
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")
    max_tokens_per_batch: Optional[int] = field(default=None)
    use_sortish_sampler: bool = field(default=True)
    sortish_chunk_factor: int = field(default=50)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="MIMIC-IV-CDM")


def infer_dim_out(model_args: ModelArguments) -> int:
    if model_args.dim_out is not None:
        return int(model_args.dim_out)
    if model_args.pretrained_path:
        config_path = os.path.join(model_args.pretrained_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if config.get("dim_out") is not None:
                return int(config["dim_out"])
    return 2048


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

    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    set_seed(training_args.seed)

    _, text_dim = load_embedding_cache(data_args.embedding_cache)

    with open(data_args.type_vocab_file, "r", encoding="utf-8") as f:
        type_vocab = json.load(f)

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

    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=infer_dim_out(model_args),
        num_classes=NUM_CLASSES,
        problem_type="single_label_classification",
    )
    model = LongTableEncoderClassifier(config=encoder_config)

    if model_args.pretrained_path:
        model = load_model_weights(model, model_args.pretrained_path, use_lora=False, is_trainable=False)

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=[m.strip() for m in model_args.lora_target_modules.split(",")],
            modules_to_save=["classifier"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        rank0_print("LoRA applied.")

    collate_fn = create_collate_fn(
        type_vocab=type_vocab,
        label_map=LABEL_MAP,
        max_table_len=data_args.max_table_len,
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

    rank0_print("Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    rank0_print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()


if __name__ == "__main__":
    main()
