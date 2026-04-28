#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

for TASK_NAME in \
  mortality \
  long_term_mortality \
  readmission \
  los_3day \
  los_7day \
  creatinine \
  bilirubin \
  platelets \
  wbc \
  final_acuity \
  imminent_discharge
do
  python train/ethos/train_eicu_ethos.py \
    --root_dir /data/EHR_data_public/eicu-crd/2.0 \
    --processed_dir /data/zikun_workspace/eicu-crd/processed \
    --train_info_path /data/zikun_workspace/eicu-crd/processed/sample_info_train.json \
    --val_info_path /data/zikun_workspace/eicu-crd/processed/sample_info_val.json \
    --vocab_dir .cache/ethos_vocab/eicu \
    --task_name "$TASK_NAME" \
    --output_dir "/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/ethos/base" \
    --report_to wandb \
    --run_name "eicu_${TASK_NAME}_ethos_base" \
    --max_seq_length 4096 \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 30 \
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
    --early_stopping_patience 10 \
    --max_train_samples 10000
done
