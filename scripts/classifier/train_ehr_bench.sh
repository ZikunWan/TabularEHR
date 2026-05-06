#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/train/Classifier
TASK_INDEX_ROOT="/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index"

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

    EPOCHS=20
    TRAIN_INFO_PATH="${TASK_INDEX_ROOT}/train/${TASK}.csv"
    VAL_INFO_PATH="${TASK_INDEX_ROOT}/val/${TASK}.csv"
    
    torchrun --nproc_per_node=$NUM_GPUS train_ehr_bench_classifier.py \
        --deepspeed "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/ds_config_zero2.json" \
        --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehr_bench/${TASK}_using_stage1_pretraining" \
        --run_name "ehr_bench_${TASK}_using_stage1_pretraining" \
        --task_name "$TASK" \
        --pretrained_path "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/contrastive_learning" \
        --train_sample_info_path "$TRAIN_INFO_PATH" \
        --val_sample_info_path "$VAL_INFO_PATH" \
        --max_table_len 4096 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs $EPOCHS \
        --learning_rate 1e-5 \
        --lr_scheduler_type "cosine" \
        --warmup_steps 100 \
        --early_stopping_patience 5 \
        --eval_strategy "steps" \
        --eval_steps 100 \
        --save_strategy "steps" \
        --save_steps 100 \
        --logging_steps 10 \
        --max_train_samples 3000 \
        --max_eval_samples 1000
done
