#!/bin/bash
set -e
source "$(dirname "$0")/../common/silent_info.sh"

deepspeed --include localhost:0,1,2,3,4,5,6,7 train/classification/train_candidate_decoder.py \
    --deepspeed "ds_config_zero2.json" \
    --dataset_name "renji" \
    --task_name "candidate_metric_prediction" \
    --data_dir "/data/EHR_data_public/Renji" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/renji/candidate_decoder" \
    --run_name "renji_candidate_decoder" \
    --max_table_len 4096 \
    --per_device_train_batch_size 16 \
    --eval_strategy no \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_candidate/renji_candidate_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --query_max_length 128 \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/joint" \
    --fine_tune_mode "full_fine_tune" \
    --num_train_epochs 50 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --save_strategy epoch \
    --save_total_limit 1 \
    --report_to wandb
