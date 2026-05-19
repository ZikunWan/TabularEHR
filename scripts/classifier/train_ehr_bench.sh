#!/bin/bash
export MIMIC_SKIP_SAMPLE_CACHE_CHECK=1
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd train/Classifier
TASK_INDEX_ROOT="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index"

TASKS=(
    "ED_Hospitalization"
    "ED_Inpatient_Mortality"
    "ED_ICU_Tranfer_12hour"
    "ED_Reattendance_3day"
    "ED_Critical_Outcomes"
    "Readmission_30day"
    "Readmission_60day"
    "Inpatient_Mortality"
    "LengthOfStay_3day"
    "LengthOfStay_7day"
    "ICU_Mortality_1day"
    "ICU_Mortality_2day"
    "ICU_Mortality_3day"
    "ICU_Mortality_7day"
    "ICU_Mortality_14day"
    "ICU_Stay_7day"
    "ICU_Stay_14day"
    "ICU_Readmission"
)

for TASK in "${TASKS[@]}"; do
    echo "==================================="
    echo "Training EHR-Bench Task: $TASK"
    echo "==================================="

    TRAIN_INFO_PATH="${TASK_INDEX_ROOT}/train/${TASK}.csv"
    VAL_INFO_PATH="${TASK_INDEX_ROOT}/val/${TASK}.csv"
    
    deepspeed --num_gpus=$NUM_GPUS train_ehr_bench_classifier.py \
        --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
        --output_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK}/table_encoder/llm_query_next_token" \
        --run_name "ehr_bench_${TASK}_llm_query_next_token" \
        --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/next_token_prediction" \
        --task_name "$TASK" \
        --train_sample_info_path "$TRAIN_INFO_PATH" \
        --val_sample_info_path "$VAL_INFO_PATH" \
        --max_table_len 16384 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 64 \
        --num_train_epochs 100 \
        --learning_rate 1e-5 \
        --early_stopping_patience 10 \
        --max_train_samples 3000 \
        --max_eval_samples 1000
done
