#!/bin/bash
set -e

cd /data/zikun_workspace/code/train/Classifier

    deepspeed --include localhost:0,1,2,3 --master_port=29500 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_clinicalbert.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_scratch_text_clinicalbert_dim2048" \
    --run_name "mimic_iv_cdm_main_diagnosis_scratch_text_clinicalbert_dim2048" \
    --dim_out 2048 \
    --max_table_len 32768 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100 &

    deepspeed --include localhost:4,5,6,7 --master_port=29501 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_pubmedbert.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_scratch_text_pubmedbert_dim2048" \
    --run_name "mimic_iv_cdm_main_diagnosis_scratch_text_pubmedbert_dim2048" \
    --dim_out 2048 \
    --max_table_len 32768 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100 &

wait

    deepspeed --include localhost:0,1,2,3 --master_port=29500 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage1.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_scratch_text_stage1_dim2048" \
    --run_name "mimic_iv_cdm_main_diagnosis_scratch_text_stage1_dim2048" \
    --dim_out 2048 \
    --max_table_len 32768 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100 &

    deepspeed --include localhost:4,5,6,7 --master_port=29501 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_scratch_text_stage2_dim2048" \
    --run_name "mimic_iv_cdm_main_diagnosis_scratch_text_stage2_dim2048" \
    --dim_out 2048 \
    --max_table_len 32768 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100 &

wait
