#!/bin/bash
set -euo pipefail

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi

OVERWRITE=true
USE_PEFT=true
FREEZE_ENCODER=false
TOKENIZER_CONFIG_PATH="/data/zikun_workspace/code/.cache/meds_encoder_tokenizers/eicu/expanded_tokenizer_config.json"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

has_training_result() {
    local output_dir="$1"
    if [ -s "$output_dir/classification_head.bin" ] && [ -s "$output_dir/sequence_classification_head_config.json" ]; then
        return 0
    fi
    return 1
}

cd /data/zikun_workspace/code/train/Llama

for TASK_NAME in \
    "mortality" \
    "long_term_mortality" \
    "readmission" \
    "los_3day" \
    "los_7day" \
    "creatinine" \
    "bilirubin" \
    "platelets" \
    "wbc" \
    "final_acuity" \
    "imminent_discharge"
do
    OUTPUT_DIR="/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr"
    RUN_NAME="eicu_${TASK_NAME}_meds_llama_base_4096_clmbr_peft"

    if has_training_result "$OUTPUT_DIR"; then
        if is_true "$OVERWRITE"; then
            echo "[OVERWRITE] Existing checkpoint found for ${TASK_NAME}, retraining: $OUTPUT_DIR"
        else
            echo "[SKIP] Existing head-only checkpoint found for ${TASK_NAME}: $OUTPUT_DIR"
            continue
        fi
    fi

    deepspeed --num_gpus="$NUM_GPUS" train_eicu_llama.py \
        --model_name_or_path "/data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr" \
        --root_dir "/data/EHR_data_public/eicu-crd/2.0" \
        --processed_dir "/data/zikun_workspace/eicu-crd/processed" \
        --train_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_train.json" \
        --val_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_val.json" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --report_to wandb \
        --overwrite_output_dir "$OVERWRITE" \
        --task_name "$TASK_NAME" \
        --tokenizer_config_path "$TOKENIZER_CONFIG_PATH" \
        --freeze_encoder "$FREEZE_ENCODER" \
        --use_peft "$USE_PEFT" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --lora_dropout "$LORA_DROPOUT" \
        --lora_target_modules "$LORA_TARGET_MODULES" \
        --max_seq_length 4096 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 128 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 30 \
        --learning_rate 2e-5 \
        --logging_steps 50 \
        --eval_strategy steps \
        --eval_steps 50 \
        --save_steps 50 \
        --save_total_limit 2 \
        --save_strategy steps \
        --bf16 True \
        --gradient_checkpointing False \
        --dataloader_num_workers 8 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
        --early_stopping_patience 10 \
        --max_train_samples 10000
done
