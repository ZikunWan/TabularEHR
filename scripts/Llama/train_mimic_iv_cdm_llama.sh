#!/bin/bash
set -euo pipefail

# ===== Editable config =====
MODEL_PATH="/data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr"
DATA_DIR="/data/EHR_data_public/mimic-iv-cdm"
CHECKPOINT_ROOT="/data/zikun_workspace/checkpoints/mimic_iv_cdm"
DEEPSPEED_CONFIG="/data/zikun_workspace/code/ds_config_zero2.json"

OVERWRITE=true
CONCEPT_MAP_DIR="/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS"
TASK_NAME="MIMIC-IV-CDM Main Disease Diagnoses"
TASK_KEY="main_diagnosis"
FREEZE_ENCODER=false
USE_PEFT=true
TOKENIZER_CONFIG_PATH="/data/zikun_workspace/code/.cache/meds_encoder_tokenizers/mimic_iv_cdm/expanded_tokenizer_config.json"
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

MAX_SEQ_LENGTH=4096
TRAIN_BATCH_SIZE=64
LEARNING_RATE=2e-4
NUM_TRAIN_EPOCHS=35
# ===========================

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="mimic_iv_cdm_meds_encoder_llama"

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

OUTPUT_DIR="${CHECKPOINT_ROOT}/${TASK_KEY}/meds_encoder/llama_base_4096_clmbr"
RUN_NAME="mimic_iv_cdm_${TASK_KEY}_meds_llama_base_4096_clmbr_peft"

if has_training_result "$OUTPUT_DIR"; then
    if is_true "$OVERWRITE"; then
        echo "[OVERWRITE] Existing checkpoint found, retraining: $OUTPUT_DIR"
    else
        echo "[SKIP] Existing head-only checkpoint found: $OUTPUT_DIR"
        exit 0
    fi
fi

export WANDB_NAME="$RUN_NAME"

EXTRA_ARGS=()
if [ -n "$CONCEPT_MAP_DIR" ]; then
    EXTRA_ARGS+=(--concept_map_dir "$CONCEPT_MAP_DIR")
fi

deepspeed --num_gpus="$NUM_GPUS" train_mimic_iv_cdm_llama.py \
    --model_name_or_path "$MODEL_PATH" \
    --root_dir "$DATA_DIR" \
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
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --logging_steps 50 \
    --eval_strategy no \
    --save_steps 50 \
    --save_total_limit 2 \
    --save_strategy steps \
    --bf16 True \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --deepspeed "$DEEPSPEED_CONFIG" \
    "${EXTRA_ARGS[@]}"
