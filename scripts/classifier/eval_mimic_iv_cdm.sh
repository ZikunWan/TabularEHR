#!/bin/bash
set -e

cd /data/zikun_workspace/code/test/Classifier

CUDA_VISIBLE_DEVICES=0 python test_mimic_iv_cdm_classifier.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/query_classifier_task_query_lora" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification" \
    --use_lora true \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --max_table_len 32768 \
    --batch_size 64
