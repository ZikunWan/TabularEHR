#!/bin/bash
set -e

cd /data/zikun_workspace/code/test/Classifier

CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/structEHR/bin/python test_mimic_iv_cdm_classifier.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_scratch_text_clinicalbert_dim2048" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_clinicalbert.pt" \
    --dim_out 2048 \
    --max_table_len 32768 \
    --batch_size 64

CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/structEHR/bin/python test_mimic_iv_cdm_classifier.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_scratch_text_pubmedbert_dim2048" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_pubmedbert.pt" \
    --dim_out 2048 \
    --max_table_len 32768 \
    --batch_size 64

CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/structEHR/bin/python test_mimic_iv_cdm_classifier.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_scratch_text_stage1_dim2048" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage1.pt" \
    --dim_out 2048 \
    --max_table_len 32768 \
    --batch_size 64

CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/structEHR/bin/python test_mimic_iv_cdm_classifier.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_scratch_text_stage2_dim2048" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --dim_out 2048 \
    --max_table_len 32768 \
    --batch_size 64
