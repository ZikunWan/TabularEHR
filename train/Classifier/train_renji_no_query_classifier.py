import glob
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.utils import logging as hf_logging

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


def rank0_print(*args, **kwargs):
    rank = os.environ.get("RANK")
    if rank is not None:
        if int(rank) == 0:
            print(*args, **kwargs)
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank in [-1, 0]:
        print(*args, **kwargs)


def quiet_non_main_process_logs():
    rank = os.environ.get("RANK")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_non_main = int(rank) != 0 if rank is not None else local_rank not in [-1, 0]
    if is_non_main:
        hf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("accelerate").setLevel(logging.ERROR)
        logging.getLogger("deepspeed").setLevel(logging.ERROR)


from dataset.renji.renji_dataset import RenjiDataset
from models.TableEncoder.config import LongTableEncoder1DConfig
from models.encoder_classifier import LongTableEncoderClassifier
from utils.collate import create_collate_fn
from utils.load_embedding import load_embedding_cache
from utils.weight_loader import load_model_weights


ACTIVE_POINTS = ["day30", "day180", "day365"]


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
    data_dir: str = field(default="/data/EHR_data_public/Renji")
    embedding_cache: str = field(default="/data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt")
    max_train_samples: Optional[int] = field(default=None)
    type_vocab_file: str = field(default="/data/zikun_workspace/code/data/type_vocab.json")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    wandb_project: Optional[str] = field(default="Renji")


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
    training_args.adam_epsilon = 1e-6
    training_args.seed = 42
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 10
    training_args.save_strategy = "steps"
    training_args.save_steps = 100
    training_args.save_total_limit = 1
    training_args.bf16 = True
    training_args.dataloader_num_workers = 32
    training_args.remove_unused_columns = False
    training_args.report_to = ["wandb"]
    training_args.save_safetensors = True
    training_args.logging_nan_inf_filter = False
    training_args.ddp_find_unused_parameters = False

    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    set_seed(training_args.seed)

    embedding_cache, text_dim = load_embedding_cache(data_args.embedding_cache)
    del embedding_cache

    vocab_path = os.path.join(project_root, data_args.type_vocab_file)
    with open(vocab_path, "r", encoding="utf-8") as f:
        type_vocab = json.load(f)

    train_dataset = RenjiDataset(
        root_dir=data_args.data_dir,
        split="train",
        table_mode="table_only",
        shuffle=True,
        max_samples=data_args.max_train_samples,
        target_prediction_points=ACTIVE_POINTS,
    )

    encoder_config = LongTableEncoder1DConfig(
        text_dim=text_dim,
        type_vocab_size=len(type_vocab),
        max_table_len=data_args.max_table_len,
        dim_out=infer_dim_out(model_args),
        num_points=len(ACTIVE_POINTS),
        num_metrics=len(RenjiDataset.ALL_METRICS),
        num_classes=len(ACTIVE_POINTS) * len(RenjiDataset.ALL_METRICS),
        problem_type="multi_label_classification",
    )
    model = LongTableEncoderClassifier(config=encoder_config)
    model = load_model_weights(model, model_args.pretrained_path, use_lora=False, is_trainable=False)

    if model_args.use_lora:
        existing_ckpts = sorted(
            glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[-1]),
        )
        if training_args.resume_from_checkpoint and existing_ckpts:
            latest_ckpt = existing_ckpts[-1]
            rank0_print(f"Loading LoRA adapter weights from checkpoint: {latest_ckpt}")
            model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=True)
        else:
            lora_cfg = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                target_modules=[m.strip() for m in model_args.lora_target_modules.split(",")],
                modules_to_save=["classifier"],
            )
            model = get_peft_model(model, lora_cfg)
        if training_args.process_index == 0:
            model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=create_collate_fn(
            type_vocab=type_vocab,
            max_table_len=data_args.max_table_len,
        ),
    )

    resume_ckpt = None
    if training_args.resume_from_checkpoint:
        rfc = training_args.resume_from_checkpoint
        if isinstance(rfc, str) and rfc.lower() not in ("true", "1", "yes"):
            resume_ckpt = rfc
        else:
            ckpt_dirs = sorted(
                glob.glob(os.path.join(training_args.output_dir, "checkpoint-*")),
                key=lambda p: int(p.rsplit("-", 1)[-1]),
            )
            if ckpt_dirs:
                resume_ckpt = ckpt_dirs[-1]

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model()


if __name__ == "__main__":
    main()
