#!/bin/bash
set -euo pipefail

NUM_GPUS=$(nvidia-smi -L | wc -l)
export TOKENIZERS_PARALLELISM=false

has_training_result() {
    local output_dir="$1"
    if [ -f "$output_dir/model.safetensors" ] || [ -f "$output_dir/pytorch_model.bin" ]; then
        return 0
    fi
    return 1
}

export WANDB_PROJECT=eicu_encoderlm

cd /data/zikun_workspace/code/train/EncoderLM

for TASK_NAME in \
    "mortality" \
    "long_term_mortality" \
    "readmission" \
    "los_3day" \
    "los_7day" \
    "final_acuity" \
    "imminent_discharge" \
    "creatinine" \
    "bilirubin" \
    "platelets" \
    "wbc"
do
    TASK_KEY=$TASK_NAME
    OUTPUT_DIR="/data/zikun_workspace/checkpoints/eicu/${TASK_KEY}/table_only/gatortron_base_2k"
    RUN_NAME="eicu_${TASK_KEY}_table_only_gatortron_base_encoderlm_finetune"

    if has_training_result "$OUTPUT_DIR"; then
        echo "[SKIP] Existing training result found for ${TASK_KEY}/gatortron_base: $OUTPUT_DIR"
        continue
    fi

    export WANDB_NAME="$RUN_NAME"

    accelerate launch --num_processes="$NUM_GPUS" train_eicu_encoderLM.py \
        --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
        --model_name_or_path /data/model_weights_public/UFNLP/gatortron-base-2k \
        --output_dir "$OUTPUT_DIR" \
        --run_name "$RUN_NAME" \
        --report_to wandb \
        --task_name "$TASK_NAME" \
        --table_mode table_only \
        --max_seq_len 2048 \
        --per_device_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs 5 \
        --learning_rate 2e-5 \
        --max_train_samples 10000 \
        --logging_steps 100 \
        --save_steps 100 \
        --save_total_limit 1 \
        --save_strategy "steps" \
        --bf16 True \
        --gradient_checkpointing False \
        --dataloader_num_workers 8 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine"
done
