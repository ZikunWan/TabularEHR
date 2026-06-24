#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
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
    
    deepspeed --num_gpus=$NUM_GPUS train/classification/train_ehrshot_classifier.py \
        --deepspeed "ds_config_zero2.json" \
        --output_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK}/after_phenotype_query_contrastive_learning" \
        --run_name "${TASK}" \
        --task_name "$TASK" \
        --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/phenotype_query_contrastive_learning" \
        --query_encoder llm \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/ehrshot_task_query_llm_embeddings.pt" \
        --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
        --max_table_len 8192 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 64 \
        --num_train_epochs 50 \
        --learning_rate 1e-5 \
        --max_train_samples 500 \
        --max_eval_samples 1000 \
        --early_stopping_patience 10
done
