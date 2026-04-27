#!/bin/bash
cd /data/zikun_workspace/code/test/Classifier

CUDA_VISIBLE_DEVICES=1 python test_mimic_iv_cdm_classifier.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/classifier_stage2" \
    --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
    --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
    --batch_size 64
