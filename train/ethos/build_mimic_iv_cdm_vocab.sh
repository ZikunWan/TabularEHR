#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python train/ethos/build_dataset_vocab.py \
  --dataset_name mimic_iv_cdm \
  --mimic_iv_cdm_root_dir /data/EHR_data_public/mimic-iv-cdm \
  --mimic_iv_cdm_concept_map_dir /data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS \
  --output_dir .cache/ethos_vocab/mimic_iv_cdm/main_disease \
  --num_workers 8 \
  --overwrite_output
