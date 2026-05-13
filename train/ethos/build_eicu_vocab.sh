#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

dataset_kwargs="["
for task_name in \
  mortality \
  long_term_mortality \
  readmission \
  los_3day \
  los_7day \
  creatinine \
  bilirubin \
  platelets \
  wbc \
  final_acuity \
  imminent_discharge
do
  dataset_kwargs+="{\"root_dir\":\"/data/EHR_data_public/eicu-crd/2.0\",\"processed_dir\":\"/data/zikun_workspace/eicu-crd/processed\",\"sample_info_path\":\"/data/zikun_workspace/eicu-crd/processed/sample_info_train.json\",\"task_name\":\"${task_name}\",\"lazy_mode\":true,\"shuffle\":false,\"table_mode\":\"table_only\",\"return_meds\":true},"
done
dataset_kwargs="${dataset_kwargs%,}]"

python train/ethos/build_dataset_vocab.py \
  --dataset dataset.eicu.eicu_dataset:EICUDataset \
  --dataset_kwargs "$dataset_kwargs" \
  --output_dir .cache/ethos_vocab/eicu \
  --num_workers 8 \
  --overwrite_output
