#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="/data/EHR_data_public/EHRSHOT"
OUTPUT_DIR=".cache/ethos_vocab/ehrshot"
NUM_WORKERS="${NUM_WORKERS:-8}"

TASKS=(
  guo_los
  guo_readmission
  guo_icu
  lab_anemia
  lab_hyperkalemia
  lab_hyponatremia
  lab_hypoglycemia
  lab_thrombocytopenia
  new_acutemi
  new_celiac
  new_hyperlipidemia
  new_hypertension
  new_lupus
  new_pancan
)

KWARGS="["
for TASK_NAME in "${TASKS[@]}"; do
  KWARGS+="{\"root_dir\":\"${DATA_DIR}\",\"sample_info_path\":\"${DATA_DIR}/index/ehrshot_train.csv\",\"task_name\":\"${TASK_NAME}\",\"lazy_mode\":true,\"table_mode\":\"table_only\",\"return_meds\":true},"
done
KWARGS="${KWARGS%,}]"

python train/ethos/build_dataset_vocab.py \
  --dataset dataset.ehrshot.ehrshot_dataset:EHRSHOTDataset \
  --dataset_kwargs "$KWARGS" \
  --output_dir "$OUTPUT_DIR" \
  --num_workers "$NUM_WORKERS" \
  --overwrite_output
