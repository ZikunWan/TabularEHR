#!/bin/bash
set -e

cd "$(dirname "$0")/../../test/Classifier"

CUDA_VISIBLE_DEVICES=0 python test_mimic_iv_cdm_classifier.py \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/phenotype_triplet_learning" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --type_vocab_file "/data/zikun_workspace/code/data/type_vocab.json" \
    --query_encoder knowledge \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/mimic_iv_cdm_task_query_knowledge_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --query_max_length 128 \
    --max_table_len 16384 \
    --batch_size 64
