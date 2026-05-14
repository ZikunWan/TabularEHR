#!/bin/bash
set -e

cd "$(dirname "$0")/../../test/Classifier"

CUDA_VISIBLE_DEVICES=0 python test_mimic_iv_cdm_classifier.py \
    --data_dir /data/EHR_data_public/mimic-iv-cdm \
    --embedding_cache /data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt \
    --checkpoint_dir /data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/query_classifier_next_token_llm_adapter_full \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --type_vocab_file /data/zikun_workspace/code/data/type_vocab.json \
    --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt \
    --query_llm_model_path /data/model_weights_public/BlueZeros/EHR-R1-1.7B \
    --max_table_len 16384 \
    --batch_size 64 \
    --pretrained_path /data/zikun_workspace/checkpoints/pretraining/next_token_prediction
