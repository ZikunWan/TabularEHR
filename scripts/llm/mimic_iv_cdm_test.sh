#!/bin/bash
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/CausalLM

NUM_GPUS=$(nvidia-smi -L | wc -l)
TABLE_MODE="table_only"
TASK_NAME="MIMIC-IV-CDM Main Disease Diagnoses"
ROOT_DIR="/home/ma-user/sfs_turbo/Data/mimic-iv-cdm"

for MODEL_KEY in \
    "qwen3_5_9b"
    #"ehr_r1_1_7b" \
    
do
    if [ "$MODEL_KEY" = "gpt_oss_20b" ]; then
        MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/mimic_iv_cdm/main_diagnosis/table_only/gpt_oss_20b"
    elif [ "$MODEL_KEY" = "medgemma_1_5_4b_it" ]; then
        MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/mimic_iv_cdm/main_diagnosis/table_only/medgemma_1_5_4b_it"
    elif [ "$MODEL_KEY" = "qwen3_5_9b" ]; then
        MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/mimic_iv_cdm/main_diagnosis/table_only/qwen3_5_9b"
    else
        MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/mimic_iv_cdm/main_diagnosis/table_only/ehr_r1_1_7b"
    fi

    if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/classification_head.bin" ]; then
        echo "Skipping $MODEL_KEY because neither model.safetensors nor classification_head.bin was found in $MODEL_PATH"
        continue
    fi

    python test_mimic_iv_cdm_llm.py \
        --model_path "$MODEL_PATH" \
        --output_dir "$MODEL_PATH" \
        --root_dir "$ROOT_DIR" \
        --task_name "$TASK_NAME" \
        --table_mode "$TABLE_MODE" \
        --max_seq_len 32768 \
        --batch_size 1 \
        --tp_size "$NUM_GPUS" \
        --use_sequence_classification True
done
