#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

for TASK_NAME in \
  Inpatient_Mortality \
  LengthOfStay_3day \
  LengthOfStay_7day \
  ICU_Mortality_1day \
  ICU_Mortality_2day \
  ICU_Mortality_3day \
  ICU_Mortality_7day \
  ICU_Mortality_14day \
  ICU_Stay_7day \
  ICU_Stay_14day \
  ICU_Readmission
do
  python train/ethos/train_ehr_bench_ethos.py \
    --data_dir /data/zikun_workspace/mimic-iv-3.1_tabular \
    --vocab_dir .cache/ethos_vocab/ehr_bench \
    --task_name "$TASK_NAME" \
    --itemid_representation code \
    --output_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/ethos/base" \
    --report_to wandb \
    --run_name "ehr_bench_${TASK_NAME}_ethos_base" \
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
    --early_stopping_patience 10 \
    --max_train_samples 3000 \
    --max_eval_samples 1000
done
