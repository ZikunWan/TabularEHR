#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false

NUM_GPUS="${TEST_NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${NUM_GPUS}" -lt 1 ]; then
    NUM_GPUS=1
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCH_CMD=(accelerate launch --num_processes "${NUM_GPUS}")
    echo "[INFO] Multi-GPU evaluation enabled (${NUM_GPUS} GPUs)."
else
    LAUNCH_CMD=(python)
    echo "[INFO] Single-GPU/CPU evaluation mode."
fi

cd /data/zikun_workspace/code/test/EncoderLM

has_eval_result() {
    local output_dir="$1"
    if [ -s "$output_dir/metrics.csv" ] && [ -s "$output_dir/raw_predictions.csv" ]; then
        return 0
    fi
    if [ -s "$output_dir/eval_logs/metrics.csv" ] && [ -s "$output_dir/eval_logs/raw_predictions.csv" ]; then
        return 0
    fi
    return 1
}

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
    MODEL_PATH="/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/table_only/gatortron_base_2k"

    if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/pytorch_model.bin" ]; then
        echo "Skipping ${TASK_NAME} because neither model.safetensors nor pytorch_model.bin was found in $MODEL_PATH"
        continue
    fi

    if has_eval_result "$MODEL_PATH"; then
        echo "[SKIP] Existing eval result found for ${TASK_NAME}/gatortron_base: $MODEL_PATH"
        continue
    fi

    "${LAUNCH_CMD[@]}" test_ehr_bench_encoderLM.py \
        --checkpoint_dir "$MODEL_PATH" \
        --output_dir "$MODEL_PATH" \
        --task_name "$TASK_NAME" \
        --table_mode table_only \
        --max_seq_len 2048 \
        --batch_size 32
done
