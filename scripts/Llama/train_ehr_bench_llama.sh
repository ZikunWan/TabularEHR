#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=ehr_bench_meds_encoder_llama

cd /data/zikun_workspace/code/train/Llama

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
  ICU_Readmission \
  ED_Hospitalization \
  ED_Inpatient_Mortality \
  ED_ICU_Tranfer_12hour \
  ED_Reattendance_3day \
  ED_Critical_Outcomes \
  Readmission_30day \
  Readmission_60day
do
  deepspeed --num_gpus="$(nvidia-smi -L 2>/dev/null | wc -l)" train_ehr_bench_llama.py \
    --model_name_or_path /data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr \
    --data_dir /data/zikun_workspace/mimic-iv-3.1_tabular \
    --output_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr" \
    --run_name "ehr_bench_${TASK_NAME}_meds_llama_base_4096_clmbr_peft" \
    --report_to wandb \
    --overwrite_output_dir true \
    --task_name "$TASK_NAME" \
    --tokenizer_config_path /data/zikun_workspace/code/.cache/meds_encoder_tokenizers/ehr_bench/expanded_tokenizer_config.json \
    --freeze_encoder false \
    --use_peft true \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --itemid_representation code \
    --max_seq_length 4096 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 50 \
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
    --dataloader_num_workers 8 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --deepspeed /data/zikun_workspace/code/ds_config_zero2.json \
    --early_stopping_patience 10 \
    --max_train_samples 3000 \
    --max_eval_samples 1000
done
