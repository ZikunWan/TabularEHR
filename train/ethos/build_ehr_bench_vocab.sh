#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

dataset_kwargs="["
for task_name in \
  Readmission_30day \
  Readmission_60day \
  Inpatient_Mortality \
  LengthOfStay_3day \
  LengthOfStay_7day \
  ICU_Mortality_1day \
  ICU_Mortality_2day \
  ICU_Mortality_3day \
  ICU_Mortality_7day \
  ICU_Mortality_14day \
  ICU_Stay_7day \
  ICU_Stay_14day \
  ICU_Readmission \
  ED_Hospitalization \
  ED_Inpatient_Mortality \
  ED_ICU_Tranfer_12hour \
  ED_Reattendance_3day \
  ED_Critical_Outcomes
do
  dataset_kwargs+="{\"root_dir\":\"/data/zikun_workspace/mimic-iv-3.1_tabular\",\"sample_info_path\":\"/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/${task_name}.csv\",\"lazy_mode\":true,\"shuffle\":false,\"table_mode\":\"table_only\",\"itemid_representation\":\"code\",\"return_meds\":true},"
done
dataset_kwargs="${dataset_kwargs%,}]"

python train/ethos/build_dataset_vocab.py \
  --dataset dataset.mimic.mimic_dataset:MIMICIV \
  --dataset_kwargs "$dataset_kwargs" \
  --output_dir .cache/ethos_vocab/ehr_bench \
  --num_workers 8 \
  --overwrite_output
