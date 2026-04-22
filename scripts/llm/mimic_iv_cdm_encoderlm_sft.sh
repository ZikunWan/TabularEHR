#!/bin/bash
set -euo pipefail

NUM_GPUS=$(nvidia-smi -L | wc -l)
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/mimic_iv_cdm/main_diagnosis/table_only/gatortron_base"
RUN_NAME="mimic_iv_cdm_main_diagnosis_table_only_gatortron_base_encoderlm_finetune"

export WANDB_PROJECT=mimic_iv_cdm_encoderlm
export WANDB_NAME="$RUN_NAME"

cd /data/zikun_workspace/code/test/EncoderLM

accelerate launch --num_processes="$NUM_GPUS" train_mimic_iv_cdm_encoderLM.py \
    --deepspeed "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/ds_config_zero2.json" \
    --model_name_or_path /home/ma-user/sfs_turbo/model_weights/UFNLP/gatortron-base \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --report_to wandb \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --table_mode table_only \
    --max_seq_len 2048 \
    --per_device_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --logging_steps 100 \
    --save_steps 500 \
    --save_total_limit 3 \
    --save_strategy "steps" \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine"
