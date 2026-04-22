#!/bin/bash
set -euo pipefail

# ===== Editable config =====
DATA_DIR="/data/EHR_data_public/mimic-iv-cdm"
CHECKPOINT_ROOT="/data/zikun_workspace/checkpoints/mimic_iv_cdm"
CONCEPT_MAP_DIR="/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS"
OVERWRITE_EVAL=true
TASK_NAME="MIMIC-IV-CDM Main Disease Diagnoses"
TASK_KEY="main_diagnosis"
MAX_SEQ_LENGTH=4096
BATCH_SIZE=64
# ===========================

export TOKENIZERS_PARALLELISM=false
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTHONWARNINGS=",ignore:pkg_resources is deprecated as an API:UserWarning:mlflow.utils.requirements_utils"

NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH_CMD=(accelerate launch --num_processes "$NUM_GPUS")
    echo "[INFO] Multi-GPU evaluation enabled (${NUM_GPUS} GPUs)."
else
    LAUNCH_CMD=(python)
    echo "[INFO] Single-GPU/CPU evaluation mode."
fi

MODEL_PATH="${CHECKPOINT_ROOT}/${TASK_KEY}/meds_encoder/llama_base_4096_clmbr"

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

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

if [ ! -s "$MODEL_PATH/classification_head.bin" ]; then
    echo "Skipping evaluation because classification_head.bin was not found in $MODEL_PATH"
    exit 0
fi

if has_eval_result "$MODEL_PATH"; then
    if is_true "$OVERWRITE_EVAL"; then
        echo "[OVERWRITE] Existing eval result found, rerunning: $MODEL_PATH"
    else
        echo "[SKIP] Existing eval result found: $MODEL_PATH"
        exit 0
    fi
fi

EXTRA_ARGS=()
if [ -n "$CONCEPT_MAP_DIR" ]; then
    EXTRA_ARGS+=(--concept_map_dir "$CONCEPT_MAP_DIR")
fi

"${LAUNCH_CMD[@]}" test_mimic_iv_cdm_llama.py \
    --checkpoint_dir "$MODEL_PATH" \
    --output_dir "$MODEL_PATH" \
    --root_dir "$DATA_DIR" \
    --task_name "$TASK_NAME" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}"
