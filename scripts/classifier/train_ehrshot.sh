#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd train/Classifier

TASKS=(
    "guo_los"
    "guo_readmission"
    "guo_icu"
    "lab_anemia"
    "lab_hyperkalemia"
    "lab_hyponatremia"
    "lab_hypoglycemia"
    "lab_thrombocytopenia"
    "new_acutemi"
    "new_celiac"
    "new_hyperlipidemia"
    "new_hypertension"
    "new_lupus"
    "new_pancan"
)

for TASK in "${TASKS[@]}"; do
    echo "==================================="
    echo "Training Task: $TASK"
    echo "==================================="
    
    deepspeed --num_gpus=$NUM_GPUS train_ehrshot_classifier.py \
        --deepspeed "ds_config_zero2.json" \
        --output_dir "/data/zikun_workspace/checkpoints/ehrshot/classifier" \
        --run_name "${TASK}_using_stage1_pretraining" \
        --task_name "$TASK" \
        --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification" \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt" \
        --query_llm_model_path "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/model_weights/BlueZeros/EHR-R1-1.7B" \
        --max_table_len 4096 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --num_train_epochs 50 \
        --learning_rate 1e-5 \
        --max_train_samples 500 \
        --max_eval_samples 1000 \
        --early_stopping_patience 10
done
