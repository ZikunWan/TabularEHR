#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

for TASK_NAME in \
  guo_los \
  guo_readmission \
  guo_icu \
  lab_anemia \
  lab_hyperkalemia \
  lab_hyponatremia \
  lab_hypoglycemia \
  lab_thrombocytopenia \
  new_acutemi \
  new_celiac \
  new_hyperlipidemia \
  new_hypertension \
  new_lupus \
  new_pancan
do
  python train/ethos/train_ehrshot_ethos.py \
    --root_dir /data/EHR_data_public/EHRSHOT \
    --train_info_path /data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv \
    --val_info_path /data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv \
    --vocab_dir .cache/ethos_vocab/ehrshot \
    --task_name "$TASK_NAME" \
    --output_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK_NAME}/ethos/base" \
    --report_to wandb \
    --run_name "ehrshot_${TASK_NAME}_ethos_base" \
    --max_seq_length 4096 \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
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
    --max_train_samples 500 \
    --max_eval_samples 1000
done
