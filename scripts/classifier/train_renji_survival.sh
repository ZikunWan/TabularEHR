#!/bin/bash
set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)
cd "$(dirname "$0")/../../train/Classifier"

deepspeed --num_gpus="$NUM_GPUS" train_renji_survival.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --output_dir "/data/zikun_workspace/checkpoints/renji/tacrolimus_survival" \
    --run_name "renji_tacrolimus_abnormal_survival" \
    --patient_subset_path "/data/zikun_workspace/code/data/patients.json" \
    --max_table_len 4096 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --monitor_fraction 0.1 \
    --monitor_seed 42 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --early_stopping_patience 10 \
    --load_best_model_at_end true \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --warmup_steps 100 \
    --bf16 true \
    --dataloader_num_workers 32 \
    --report_to wandb \
    --query_encoder knowledge \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/renji_survival_task_query_knowledge_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --query_max_length 128 \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/phenotype_metric_learning"
