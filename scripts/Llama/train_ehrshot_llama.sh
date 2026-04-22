#!/bin/bash
set -euo pipefail

NUM_GPUS="${TRAIN_NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${NUM_GPUS}" -lt 1 ]; then
    NUM_GPUS=1
fi

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-ehrshot_meds_encoder_llama}"
OVERWRITE="${OVERWRITE:-true}"

MODEL_PATH="${MODEL_PATH:-/data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/zikun_workspace/checkpoints/ehrshot}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-/data/zikun_workspace/code/ds_config_zero2.json}"

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
    "guo_los" \
    "guo_readmission" \
    "guo_icu" \
    "lab_anemia" \
    "lab_hyperkalemia" \
    "lab_hyponatremia" \
    "lab_hypoglycemia" \
    "lab_thrombocytopenia" \
    "new_acutemi" \
    "new_celiac" \
    "new_hyperlipidemia" \
    "new_hypertension" \
    "new_lupus" \
    "new_pancan"
do
    OUTPUT_DIR="${CHECKPOINT_ROOT}/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr"
    RUN_NAME="ehrshot_${TASK_NAME}_meds_llama_base_4096_clmbr_head_only"

    if has_training_result "$OUTPUT_DIR"; then
        if is_true "$OVERWRITE"; then
            echo "[OVERWRITE] Existing checkpoint found for ${TASK_NAME}, retraining: $OUTPUT_DIR"
        else
            echo "[SKIP] Existing head-only checkpoint found for ${TASK_NAME}: $OUTPUT_DIR"
            continue
        fi
    fi

    export WANDB_NAME="$RUN_NAME"

    deepspeed --num_gpus="$NUM_GPUS" train_ehrshot_llama.py \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --report_to wandb \
        --overwrite_output_dir "$OVERWRITE" \
        --task_name "$TASK_NAME" \
        --freeze_encoder True \
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
        --greater_is_better True \
        --save_steps 50 \
        --save_total_limit 2 \
        --save_strategy steps \
        --bf16 True \
        --gradient_checkpointing False \
        --dataloader_num_workers 16 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --max_train_samples 500 \
        --max_eval_samples 1000 \
        --deepspeed "$DEEPSPEED_CONFIG" \
        --early_stopping_patience 10
done
