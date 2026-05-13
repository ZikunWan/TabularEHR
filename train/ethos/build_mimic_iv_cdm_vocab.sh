#!/bin/bash
set -euo pipefail

python train/ethos/build_dataset_vocab.py \
  --dataset dataset.mimic_iv_cdm.mimic_iv_cdm_dataset:MIMICIVCDM \
  --dataset_kwargs "{\"root_dir\":\"/data/EHR_data_public/mimic-iv-cdm\",\"split\":\"train\",\"lazy_mode\":true,\"shuffle\":false,\"table_mode\":\"table_only\",\"task_name\":\"MIMIC-IV-CDM Main Disease Diagnoses\",\"return_meds\":true,\"concept_map_dir\":\"/data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS\"}" \
  --output_dir .cache/ethos_vocab/mimic_iv_cdm/main_disease \
  --num_workers 8 \
  --overwrite_output
