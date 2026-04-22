#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/Classifier

TASKS=(
    #"guo_los"
    #"guo_readmission"
    #"guo_icu"
    #"lab_anemia"
    #"lab_hyperkalemia"
    #"lab_hyponatremia"
    "lab_hypoglycemia"
    #"lab_thrombocytopenia"
    #"new_acutemi"
    #"new_celiac"
    #"new_hyperlipidemia"
    #"new_hypertension"
    #"new_lupus"
    #"new_pancan"
)

# Base directory where stage1-pretrained fine-tuning saved task checkpoints
BASE_CHECKPOINT_DIR="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehrshot/using_stage1_pretraining"

for TASK in "${TASKS[@]}"; do
    echo "==================================="
    echo "Testing Task: $TASK"
    echo "==================================="
    
    TASK_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/${TASK}"
    if [ ! -d "$TASK_CHECKPOINT_DIR" ]; then
        echo "Warning: Checkpoint directory $TASK_CHECKPOINT_DIR does not exist. Skipping $TASK."
        continue
    fi
    if [ ! -f "${TASK_CHECKPOINT_DIR}/model.safetensors" ] && [ ! -f "${TASK_CHECKPOINT_DIR}/pytorch_model.bin" ]; then
        echo "Warning: No final model file found in $TASK_CHECKPOINT_DIR. Skipping $TASK."
        continue
    fi
    python test_ehrshot_classifier.py \
        --checkpoint_dir "$TASK_CHECKPOINT_DIR" \
        --task_name "$TASK" \
        --batch_size 8 \
        --max_eval_samples 1000
        
    echo "Finished testing ${TASK}."
    echo ""
done
