#!/bin/bash
set -e
source "$(dirname "$0")/../common/silent_info.sh"

deepspeed --include localhost:4,5,6,7 train/classification/train_candidate_decoder.py \
    --deepspeed "ds_config_zero2.json" \
    --dataset_name "mimic_iv_cdm" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/candidate_decoder" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/joint" \
    --fine_tune_mode "full_fine_tune" \
    --run_name "mimic_iv_cdm_candidate_decoder" \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_candidate/mimic_iv_cdm_task_candidate_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --max_table_len 16384 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}'
