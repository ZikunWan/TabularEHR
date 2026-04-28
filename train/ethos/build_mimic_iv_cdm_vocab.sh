#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_DIR="/data/EHR_data_public/mimic-iv-cdm"
CONCEPT_MAP_DIR="/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS"
TASK_NAME="MIMIC-IV-CDM Main Disease Diagnoses"
TASK_KEY="main_disease"
OUTPUT_DIR=".cache/ethos_vocab/mimic_iv_cdm/${TASK_KEY}"
NUM_WORKERS="${NUM_WORKERS:-8}"

python train/ethos/build_dataset_vocab.py \
  --dataset dataset.mimic_iv_cdm.mimic_iv_cdm_dataset:MIMICIVCDM \
  --dataset_kwargs "{\"root_dir\":\"${DATA_DIR}\",\"split\":\"train\",\"lazy_mode\":true,\"shuffle\":false,\"table_mode\":\"table_only\",\"task_name\":\"${TASK_NAME}\",\"return_meds\":true,\"concept_map_dir\":\"${CONCEPT_MAP_DIR}\"}" \
  --output_dir "$OUTPUT_DIR" \
  --num_workers "$NUM_WORKERS" \
  --overwrite_output
