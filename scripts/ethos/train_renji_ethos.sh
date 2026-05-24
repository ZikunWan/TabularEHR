#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

python train/ethos/train_renji_ethos.py \
  --root_dir /data/EHR_data_public/Renji \
  --vocab_dir .cache/ethos_vocab/renji \
  --task_name multi_label_prediction \
  --eval_split all_valid \
  --output_dir /data/zikun_workspace/checkpoints/renji/ethos/base \
  --report_to wandb \
  --run_name renji_ethos_base \
  --max_seq_length 4096 \
  --n_layer 6 \
  --n_head 8 \
  --n_embd 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 50 \
  --learning_rate 2e-4 \
  --logging_steps 50 \
  --eval_strategy steps \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --save_strategy steps \
  --dataloader_num_workers 8 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --early_stopping_patience 10
