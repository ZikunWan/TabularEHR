#!/bin/bash
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/EncoderLM

TABLE_MODE="table_only"
TASK_NAME="MIMIC-IV-CDM Main Disease Diagnoses"
ROOT_DIR="/home/ma-user/sfs_turbo/Data/mimic-iv-cdm"

python test_mimic_iv_cdm_encoderLM.py \
    --checkpoint_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/mimic_iv_cdm/main_diagnosis/table_only/gatortron_base" \
    --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/mimic_iv_cdm/main_diagnosis/table_only/gatortron_base" \
    --root_dir "$ROOT_DIR" \
    --task_name "$TASK_NAME" \
    --table_mode "$TABLE_MODE" \
    --max_seq_len 2048 \
    --batch_size 1

