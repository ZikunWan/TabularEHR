#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

ROOT_DIR="/data/EHR_data_public/eicu-crd/2.0"
PROCESSED_DIR="/data/zikun_workspace/eicu-crd/processed"
OUTPUT_DIR=".cache/ethos_vocab/eicu"
NUM_WORKERS="${NUM_WORKERS:-8}"

TASKS=(
  mortality
  long_term_mortality
  readmission
  los_3day
  los_7day
  creatinine
  bilirubin
  platelets
  wbc
  final_acuity
  imminent_discharge
)

KWARGS="["
for TASK_NAME in "${TASKS[@]}"; do
  KWARGS+="{\"root_dir\":\"${ROOT_DIR}\",\"processed_dir\":\"${PROCESSED_DIR}\",\"sample_info_path\":\"${PROCESSED_DIR}/sample_info_train.json\",\"task_name\":\"${TASK_NAME}\",\"lazy_mode\":true,\"shuffle\":false,\"table_mode\":\"table_only\",\"return_meds\":true},"
done
KWARGS="${KWARGS%,}]"

python train/ethos/build_dataset_vocab.py \
  --dataset dataset.eicu.eicu_dataset:EICUDataset \
  --dataset_kwargs "$KWARGS" \
  --output_dir "$OUTPUT_DIR" \
  --num_workers "$NUM_WORKERS" \
  --overwrite_output
