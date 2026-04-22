#!/bin/bash
set -euo pipefail

# ===== Editable config =====
DATA_DIR="/data/zikun_workspace/mimic-iv-3.1_tabular"
CHECKPOINT_ROOT="/data/zikun_workspace/checkpoints/ehr_bench"
ITEMID_REPRESENTATION="code"
CONCEPT_MAP_DIR=""
OVERWRITE_EVAL=true
MAX_SEQ_LENGTH=4096
BATCH_SIZE=64
# ===========================

export TOKENIZERS_PARALLELISM=false
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTHONWARNINGS=",ignore:pkg_resources is deprecated as an API:UserWarning:mlflow.utils.requirements_utils"

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
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
    #"ED_Hospitalization" \
    #"ED_Inpatient_Mortality" \
    #"ED_ICU_Tranfer_12hour" \
    #"ED_Reattendance_3day" \
    #"ED_Critical_Outcomes" \
    
do
    MODEL_PATH="${CHECKPOINT_ROOT}/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr"

    if [ ! -s "$MODEL_PATH/classification_head.bin" ]; then
        echo "Skipping ${TASK_NAME} because classification_head.bin was not found in $MODEL_PATH"
        continue
    fi

    if has_eval_result "$MODEL_PATH"; then
        if [ "$OVERWRITE_EVAL" = true ]; then
            echo "[OVERWRITE] Existing eval result found for ${TASK_NAME}, rerunning: $MODEL_PATH"
        else
            echo "[SKIP] Existing eval result found for ${TASK_NAME}: $MODEL_PATH"
            continue
        fi
    fi

    "${LAUNCH_CMD[@]}" test_ehr_bench_llama.py \
        --checkpoint_dir "$MODEL_PATH" \
        --output_dir "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --task_name "$TASK_NAME" \
        --itemid_representation "$ITEMID_REPRESENTATION" \
        --max_seq_length "$MAX_SEQ_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --concept_map_dir "$CONCEPT_MAP_DIR"
done
