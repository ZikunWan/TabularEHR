#!/bin/bash
cd /data/zikun_workspace/code/test

NUM_GPUS=$(nvidia-smi -L | wc -l)
TABLE_MODE="table_only"

has_valid_model_files() {
    local model_dir="$1"

    if [ -f "$model_dir/model.safetensors" ] || [ -f "$model_dir/pytorch_model.bin" ] || [ -f "$model_dir/adapter_config.json" ]; then
        return 0
    fi

    # Sequence-classification checkpoints in this repo are saved as
    # classification head + metadata that points to base model weights.
    if [ -f "$model_dir/classification_head.bin" ] && [ -f "$model_dir/sequence_classification_head_config.json" ]; then
        return 0
    fi

    return 1
}

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
    TASK_KEY=$TASK_NAME

    for MODEL_KEY in \
        "qwen3_5_9b" \
        "gpt_oss_20b" \
        "ehr_r1_1_7b" \
        "medgemma_1_5_4b_it"
    do
        MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/eicu/${TASK_KEY}/${TABLE_MODE}/${MODEL_KEY}"
        EVAL_MODEL_PATH="$MODEL_PATH"

        if ! has_valid_model_files "$EVAL_MODEL_PATH"; then
            LATEST_CHECKPOINT=$(find "$MODEL_PATH" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$LATEST_CHECKPOINT" ] && has_valid_model_files "$LATEST_CHECKPOINT"; then
                EVAL_MODEL_PATH="$LATEST_CHECKPOINT"
            else
                echo "Skipping $MODEL_KEY because valid model files were not found in $MODEL_PATH"
                continue
            fi
        fi

        echo "Evaluating $MODEL_KEY using $EVAL_MODEL_PATH"

        python test_eicu_llm.py \
            --model_path "$EVAL_MODEL_PATH" \
            --output_dir "$MODEL_PATH" \
            --task_name "$TASK_NAME" \
            --max_seq_len 32768 \
            --tp_size "$NUM_GPUS" \
            --use_sequence_classification True \
            --table_mode "$TABLE_MODE"
    done
done
