#!/bin/bash
set -e

CUDA_VISIBLE_DEVICES=0 python test/classification/test_candidate_decoder.py \
    --dataset_name renji \
    --task_name candidate_metric_prediction \
    --data_dir /data/EHR_data_public/Renji \
    --embedding_cache /data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt \
    --checkpoint_dir /data/zikun_workspace/checkpoints/renji/candidate_decoder \
    --split test \
    --type_vocab_file data/type_vocab.json \
    --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_candidate/renji_candidate_embeddings.pt \
    --knowledge_encoder_path /data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt \
    --knowledge_encoder_base_model_path /data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT \
    --query_max_length 128 \
    --max_table_len 4096 \
    --batch_size 256
