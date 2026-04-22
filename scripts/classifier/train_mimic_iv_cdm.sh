#!/bin/bash

PRETRAINED_PATH_1="/data/zikun_workspace/checkpoints/contrastive_learning/tabular_encoder/model.safetensors"
PRETRAINED_PATH_2="/data/zikun_workspace/checkpoints/pretraining/stage2_bi_reconstruct/tabular_encoder/model.safetensors"

cd /data/zikun_workspace/code/train/Classifier

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29500 --nproc_per_node=4 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_stage1" \
    --run_name "mimic_iv_cdm_main_diagnosis_classifier_stage1" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --per_device_train_batch_size 64 \
    --num_train_epochs 100 \
    --learning_rate 5e-4 \
    --pretrained_path "$PRETRAINED_PATH_1" &

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29501 --nproc_per_node=4 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_stage2" \
    --run_name "mimic_iv_cdm_main_diagnosis_classifier_stage2" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --per_device_train_batch_size 64 \
    --num_train_epochs 100 \
    --learning_rate 5e-4 \
    --pretrained_path "$PRETRAINED_PATH_2" &

wait
