#!/bin/bash
set -euo pipefail

dataset_kwargs="["
for task_name in \
  guo_los \
  guo_readmission \
  guo_icu \
  lab_anemia \
  lab_hyperkalemia \
  lab_hyponatremia \
  lab_hypoglycemia \
  lab_thrombocytopenia \
  new_acutemi \
  new_celiac \
  new_hyperlipidemia \
  new_hypertension \
  new_lupus \
  new_pancan
do
  dataset_kwargs+="{\"root_dir\":\"/data/EHR_data_public/EHRSHOT\",\"sample_info_path\":\"/data/EHR_data_public/EHRSHOT/index/ehrshot_train.csv\",\"task_name\":\"${task_name}\",\"lazy_mode\":true,\"table_mode\":\"table_only\",\"return_meds\":true},"
done
dataset_kwargs="${dataset_kwargs%,}]"

python train/ethos/build_dataset_vocab.py \
  --dataset dataset.ehrshot.ehrshot_dataset:EHRSHOTDataset \
  --dataset_kwargs "$dataset_kwargs" \
  --output_dir .cache/ethos_vocab/ehrshot \
  --num_workers 8 \
  --overwrite_output
