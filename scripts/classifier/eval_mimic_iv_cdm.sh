#!/bin/bash
set -e

cd "$(dirname "$0")/../../test/Classifier"

CUDA_VISIBLE_DEVICES=0 python test_mimic_iv_cdm_classifier.py \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/phenotype_metric_learning" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --type_vocab_file "/data/zikun_workspace/code/data/type_vocab.json" \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/renji_task_query_knowledge_embeddings.pt" \
    --max_table_len 16384 \
    --batch_size 64
