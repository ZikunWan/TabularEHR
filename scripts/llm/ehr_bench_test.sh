#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
TABLE_MODE="table_only"
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/CausalLM

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

    for MODEL_KEY in \
        "qwen3_5_9b" \
        "gpt_oss_20b" \
        "medgemma_1_5_4b_it" \
        "ehr_r1_1_7b"
    do
        MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehr_bench/${TASK_KEY}/${TABLE_MODE}/${MODEL_KEY}"

        if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/classification_head.bin" ]; then
            echo "Skipping $MODEL_KEY/$TASK_KEY because neither model.safetensors nor classification_head.bin was found in $MODEL_PATH"
            continue
        fi

        python test_ehr_bench_llm.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$MODEL_PATH" \
            --task_name "$TASK_NAME" \
            --table_mode "$TABLE_MODE" \
            --max_seq_len 8192 \
            --batch_size 1 \
            --tp_size "$NUM_GPUS" \
            --use_sequence_classification True
    done
done
