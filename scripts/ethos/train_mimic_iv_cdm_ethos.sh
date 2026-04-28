#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

python train/ethos/train_mimic_iv_cdm_ethos.py \
  --root_dir /data/EHR_data_public/mimic-iv-cdm \
  --concept_map_dir /data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS \
  --vocab_dir .cache/ethos_vocab/mimic_iv_cdm/main_disease \
  --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
  --output_dir /data/zikun_workspace/checkpoints/mimic_iv_cdm/main_disease/ethos/base \
  --report_to wandb \
  --run_name mimic_iv_cdm_main_disease_ethos_base \
  --max_seq_length 4096 \
  --n_layer 6 \
  --n_head 8 \
  --n_embd 512 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 35 \
  --learning_rate 2e-4 \
  --logging_steps 50 \
  --eval_strategy no \
  --save_steps 50 \
  --save_total_limit 2 \
  --save_strategy steps \
  --dataloader_num_workers 8 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine
