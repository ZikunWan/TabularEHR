#!/bin/bash
set -e
source "$(dirname "$0")/../common/silent_info.sh"

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
    deepspeed --include localhost:4,5,6,7 train/classification/train_candidate_decoder.py \
        --deepspeed "ds_config_zero2.json" \
        --dataset_name "ehrshot" \
        --output_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK}/candidate_decoder" \
        --run_name "${TASK}_candidate_decoder" \
        --task_name "${TASK}" \
        --data_dir "/data/EHR_data_public/EHRSHOT" \
        --train_info_path "/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv" \
        --val_info_path "/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv" \
        --embedding_cache "/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt" \
        --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/joint" \
        --fine_tune_mode "full_fine_tune" \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_candidate/ehrshot_task_candidate_embeddings.pt" \
        --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
        --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --max_table_len 4096 \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 64 \
        --num_train_epochs 50 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine_with_min_lr \
        --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
        --max_train_samples 500 \
        --max_eval_samples 1000 \
        --early_stopping_patience 10
done
