#!/bin/bash
set -e

cd "$(dirname "$0")/../../test/Classifier"

CUDA_VISIBLE_DEVICES=0 python test_renji_classifier.py \
    --data_dir /data/EHR_data_public/Renji \
    --embedding_cache /data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt \
    --checkpoint_dir /data/zikun_workspace/checkpoints/renji/after_contrastive_learning \
    --split test \
    --type_vocab_file /data/zikun_workspace/code/data/type_vocab.json \
    --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt \
    --query_llm_model_path /data/model_weights_public/BlueZeros/EHR-R1-1.7B \
    --max_table_len 16384 \
    --batch_size 32
