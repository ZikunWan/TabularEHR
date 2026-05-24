#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=renji_meds_encoder_llama

cd /data/zikun_workspace/code/train/Llama

deepspeed --num_gpus="$(nvidia-smi -L 2>/dev/null | wc -l)" train_renji_llama.py \
  --model_name_or_path /data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr \
  --tokenizer_config_path /data/zikun_workspace/code/.cache/meds_encoder_tokenizers/renji/expanded_tokenizer_config.json \
  --root_dir /data/EHR_data_public/Renji \
  --target_prediction_points day0,day30,day180,day365 \
  --output_dir /data/zikun_workspace/checkpoints/renji/meds_encoder/llama_base_4096_clmbr \
  --run_name renji_meds_llama_base_4096_clmbr_peft \
  --report_to wandb \
  --overwrite_output_dir true \
  --freeze_encoder false \
  --use_peft true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  --max_seq_length 4096 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 30 \
  --learning_rate 2e-5 \
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
  --deepspeed /data/zikun_workspace/code/ds_config_zero2.json
