#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/train/Classifier

TASKS=(
    #"guo_los"
    #"guo_readmission"
    #"guo_icu"
    #"lab_anemia"
    "lab_hyperkalemia"
    #"lab_hyponatremia"
    #"lab_hypoglycemia"
    #"lab_thrombocytopenia"
    #"new_acutemi"
    #"new_celiac"
    #"new_hyperlipidemia"
    #"new_hypertension"
    #"new_lupus"
    #"new_pancan"
)

PRETRAINED_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/contrastive_learning"

for TASK in "${TASKS[@]}"; do
    echo "==================================="
    echo "Training Task: $TASK"
    echo "==================================="
    
    torchrun --nproc_per_node=$NUM_GPUS train_ehrshot_classifier.py \
        --deepspeed "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/ds_config_zero2.json" \
        --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehrshot/using_stage1_pretraining" \
        --run_name "${TASK}_using_stage1_pretraining" \
        --task_name "$TASK" \
        --pretrained_path "$PRETRAINED_PATH" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --per_device_eval_batch_size 2 \
        --num_train_epochs 50 \
        --learning_rate 1e-5 \
        --max_train_samples 500 \
        --max_eval_samples 1000 \
        --early_stopping_patience 10
done
