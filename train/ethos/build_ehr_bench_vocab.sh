#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="/data/zikun_workspace/mimic-iv-3.1_tabular"
OUTPUT_ROOT=".cache/ethos_tokenized/ehr_bench"
NUM_WORKERS="${NUM_WORKERS:-8}"

TASKS=(
  Readmission_30day
  Readmission_60day
  Inpatient_Mortality
  LengthOfStay_3day
  LengthOfStay_7day
  ICU_Mortality_1day
  ICU_Mortality_2day
  ICU_Mortality_3day
  ICU_Mortality_7day
  ICU_Mortality_14day
  ICU_Stay_7day
  ICU_Stay_14day
  ICU_Readmission
  ED_Hospitalization
  ED_Inpatient_Mortality
  ED_ICU_Tranfer_12hour
  ED_Reattendance_3day
  ED_Critical_Outcomes
)

for SPLIT in train val test; do
  KWARGS="["
  for TASK_NAME in "${TASKS[@]}"; do
    KWARGS+="{\"root_dir\":\"${DATA_DIR}\",\"sample_info_path\":\"${DATA_DIR}/task_index/${SPLIT}/${TASK_NAME}.csv\",\"lazy_mode\":true,\"shuffle\":false,\"table_mode\":\"table_only\",\"itemid_representation\":\"code\",\"return_meds\":true},"
  done
  KWARGS="${KWARGS%,}]"

  CMD=(python train/ethos/build_dataset_vocab.py
    --dataset dataset.mimic.mimic_dataset:MIMICIV
    --dataset_kwargs "$KWARGS"
    --output_dir "${OUTPUT_ROOT}/${SPLIT}"
    --num_workers "$NUM_WORKERS"
    --overwrite_output)

  if [ "$SPLIT" != "train" ]; then
    CMD+=(--reuse_artifacts_dir "${OUTPUT_ROOT}/train")
  fi

  "${CMD[@]}"
done
