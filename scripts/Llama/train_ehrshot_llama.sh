#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=ehrshot_meds_encoder_llama

cd /data/zikun_workspace/code/train/Llama

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
  deepspeed --num_gpus="$(nvidia-smi -L 2>/dev/null | wc -l)" train_ehrshot_llama.py \
    --model_name_or_path /data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr \
    --output_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr" \
    --run_name "ehrshot_${TASK_NAME}_meds_llama_base_4096_clmbr_head_only" \
    --report_to wandb \
    --overwrite_output_dir true \
    --task_name "$TASK_NAME" \
    --freeze_encoder true \
    --max_seq_length 4096 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 30 \
    --learning_rate 2e-5 \
    --logging_steps 50 \
    --eval_strategy steps \
    --eval_steps 50 \
    --metric_for_best_model eval_auroc \
    --greater_is_better true \
    --save_steps 50 \
    --save_total_limit 2 \
    --save_strategy steps \
    --bf16 true \
    --gradient_checkpointing false \
    --dataloader_num_workers 16 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --max_train_samples 500 \
    --max_eval_samples 1000 \
    --deepspeed /data/zikun_workspace/code/ds_config_zero2.json \
    --early_stopping_patience 10
done
