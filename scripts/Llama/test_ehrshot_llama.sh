#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTHONWARNINGS="${PYTHONWARNINGS:-},ignore:pkg_resources is deprecated as an API:UserWarning:mlflow.utils.requirements_utils"

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

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/zikun_workspace/checkpoints/ehrshot}"

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

cd /data/zikun_workspace/code/test/Llama

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
    MODEL_PATH="${CHECKPOINT_ROOT}/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr"

    if [ ! -s "$MODEL_PATH/classification_head.bin" ]; then
        echo "Skipping ${TASK_NAME} because classification_head.bin was not found in $MODEL_PATH"
        continue
    fi

    if has_eval_result "$MODEL_PATH"; then
        echo "[SKIP] Existing eval result found for ${TASK_NAME}: $MODEL_PATH"
        continue
    fi

    "${LAUNCH_CMD[@]}" test_ehrshot_llama.py \
        --checkpoint_dir "$MODEL_PATH" \
        --output_dir "$MODEL_PATH" \
        --task_name "$TASK_NAME" \
        --max_seq_length 4096 \
        --batch_size 32 \
        --max_test_samples 1000
done
