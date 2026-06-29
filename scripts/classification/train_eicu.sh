#!/bin/bash
set -e
source "$(dirname "$0")/../common/silent_info.sh"

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
    deepspeed --include localhost:4,5,6,7 train/classification/train_candidate_decoder.py \
        --deepspeed "ds_config_zero2.json" \
        --dataset_name "eicu" \
        --output_dir "/data/zikun_workspace/checkpoints/eicu/candidate_decoder/${TASK}" \
        --run_name "eicu_${TASK}_candidate_decoder" \
        --task_name "${TASK}" \
        --data_dir "/data/EHR_data_public/eicu-crd/2.0" \
        --processed_dir "/data/zikun_workspace/eicu-crd/processed" \
        --train_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_train.json" \
        --val_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_val.json" \
        --embedding_cache "/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt" \
        --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/joint" \
        --fine_tune_mode "full_fine_tune" \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_candidate/eicu_task_candidate_embeddings.pt" \
        --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
        --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --max_table_len 16384 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --num_train_epochs 50 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine_with_min_lr \
        --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
        --max_train_samples 10000
done
