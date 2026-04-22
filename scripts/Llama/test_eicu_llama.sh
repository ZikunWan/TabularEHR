#!/bin/bash
set -euo pipefail

OVERWRITE_EVAL=true

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
    MODEL_PATH="/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr"

    if [ ! -s "$MODEL_PATH/classification_head.bin" ]; then
        echo "[SKIP] classification_head.bin not found for ${TASK_NAME}: $MODEL_PATH"
        continue
    fi

    if has_eval_result "$MODEL_PATH"; then
        if is_true "$OVERWRITE_EVAL"; then
            echo "[OVERWRITE] Existing eval result found, rerunning: $MODEL_PATH"
        else
            echo "[SKIP] Existing eval result found: $MODEL_PATH"
            continue
        fi
    fi

    python test_eicu_llama.py \
        --checkpoint_dir "$MODEL_PATH" \
        --task_name "$TASK_NAME" \
        --max_seq_length 4096 \
        --batch_size 16 \
        --output_dir "$MODEL_PATH/eval_logs" \
        --root_dir "/data/EHR_data_public/eicu-crd/2.0" \
        --processed_dir "/data/zikun_workspace/eicu-crd/processed"
done
