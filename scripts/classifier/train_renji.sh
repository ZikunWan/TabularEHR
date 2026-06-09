#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd train/Classifier

deepspeed --num_gpus=$NUM_GPUS train_renji_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --output_dir "/data/zikun_workspace/checkpoints/renji/phenotype_metric_learning" \
    --run_name "renji_phenotype_metric_learning" \
    --max_table_len 4096 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --monitor_fraction 0.1 \
    --monitor_seed 42 \
    --eval_strategy steps \
    --eval_steps 100 \
    --num_train_epochs 100 \
    --learning_rate 3e-5 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
    --warmup_steps 100 \
    --query_encoder knowledge \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/renji_task_query_knowledge_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --query_max_length 128 \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/phenotype_metric_learning"
