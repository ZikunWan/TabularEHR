#!/bin/bash
NUM_GPUS=8
cd /home/ma-user/sfs_turbo/sai6/zkwan/code/train/Classifier

TASKS=(
    "mortality"
    "long_term_mortality"
    "readmission"
    "los_3day"
    "los_7day"
    "creatinine"
    "bilirubin"
    "platelets"
    "wbc"
    "final_acuity"
    "imminent_discharge"
)

for TASK in "${TASKS[@]}"; do
    echo "==================================="
    echo "Training Task: $TASK"
    echo "==================================="
    
    EPOCHS=20
    
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=$NUM_GPUS train_eicu_classifier.py \
        --deepspeed "/home/ma-user/sfs_turbo/sai6/zkwan/code/ds_config_zero2.json" \
        --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/eicu/${TASK}" \
        --run_name "eicu_${TASK}" \
        --task_name "$TASK" \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 128 \
        --num_train_epochs $EPOCHS \
        --learning_rate 1e-4 \
        --max_train_samples 10000
done
