#!/bin/bash
set -e

cd /data/zikun_workspace/code/train/Classifier

deepspeed --include localhost:4,5,6,7 --master_port=29501 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/query_classifier_task_query_lora" \
    --run_name "mimic_iv_cdm_main_diagnosis_query_classifier_task_query_lora" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification" \
    --use_lora true \
    --max_table_len 32768 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5
