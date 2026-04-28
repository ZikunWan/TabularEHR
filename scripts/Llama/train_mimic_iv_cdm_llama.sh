#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=mimic_iv_cdm_meds_encoder_llama

cd /data/zikun_workspace/code/train/Llama

deepspeed --num_gpus="$(nvidia-smi -L 2>/dev/null | wc -l)" train_mimic_iv_cdm_llama.py \
  --model_name_or_path /data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr \
  --root_dir /data/EHR_data_public/mimic-iv-cdm \
  --output_dir /data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/meds_encoder/llama_base_4096_clmbr \
  --run_name mimic_iv_cdm_main_diagnosis_meds_llama_base_4096_clmbr_peft \
  --report_to wandb \
  --overwrite_output_dir true \
  --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
  --tokenizer_config_path /data/zikun_workspace/code/.cache/meds_encoder_tokenizers/mimic_iv_cdm/expanded_tokenizer_config.json \
  --freeze_encoder false \
  --use_peft true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max_seq_length 4096 \
  --per_device_train_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 35 \
  --learning_rate 2e-4 \
  --logging_steps 50 \
  --eval_strategy no \
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
  --concept_map_dir /data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS
