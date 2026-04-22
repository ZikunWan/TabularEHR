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

export WANDB_PROJECT=ehr_bench_encoderlm

cd /data/zikun_workspace/code/train/EncoderLM

for TASK_NAME in \
    "ED_Hospitalization" \
    "ED_Inpatient_Mortality" \
    "ED_ICU_Tranfer_12hour" \
    "ED_Reattendance_3day" \
    "ED_Critical_Outcomes" \
    "Readmission_30day" \
    "Readmission_60day" \
    "Inpatient_Mortality" \
    "LengthOfStay_3day" \
    "LengthOfStay_7day" \
    "ICU_Mortality_1day" \
    "ICU_Mortality_2day" \
    "ICU_Mortality_3day" \
    "ICU_Mortality_7day" \
    "ICU_Mortality_14day" \
    "ICU_Stay_7day" \
    "ICU_Stay_14day" \
    "ICU_Readmission"
do
    TASK_KEY=$TASK_NAME
    OUTPUT_DIR="/data/zikun_workspace/checkpoints/ehr_bench/${TASK_KEY}/table_only/gatortron_base_2k"
    RUN_NAME="ehr_bench_${TASK_KEY}_table_only_gatortron_base_encoderlm_finetune"

    if has_training_result "$OUTPUT_DIR"; then
        echo "[SKIP] Existing training result found for ${TASK_KEY}/gatortron_base: $OUTPUT_DIR"
        continue
    fi

    export WANDB_NAME="$RUN_NAME"

    accelerate launch --num_processes="$NUM_GPUS" train_ehr_bench_encoderLM.py \
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
        --max_train_samples 3000 \
        --logging_steps 100 \
        --save_steps 200 \
        --save_total_limit 3 \
        --save_strategy "steps" \
        --bf16 True \
        --gradient_checkpointing False \
        --dataloader_num_workers 8 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine"
done
