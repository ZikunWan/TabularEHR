#!/bin/bash
set -euo pipefail

python ./preprocess/build_tte_task_index.py \
    --death_only \
    --death_horizon_days 3650 \
    --output_dir "/data/zikun_workspace/tte_death_task_index" \
    --mimic_ehr_dir "/data/zikun_workspace/mimic-iv-3.1_tabular/patients_ehr" \
    --mimic_train_index_dir "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train" \
    --mimic_val_index_dir "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val" \
    --eicu_train_sample_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_train.json" \
    --eicu_val_sample_info_path "/data/zikun_workspace/eicu-crd/processed/sample_info_val.json" \
    --eicu_cohorts_path "/data/zikun_workspace/eicu-crd/processed/cohorts.csv" \
    --ehrshot_root_dir "/data/EHR_data_public/EHRSHOT" \
    --ehrshot_train_index_path "/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv" \
    --ehrshot_val_index_path "/data/EHR_data_public/EHRSHOT/index/ehrshot_val.csv" \
    --num_workers 32 \
    --worker_chunksize 32
