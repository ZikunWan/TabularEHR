#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python train/ethos/build_dataset_vocab.py \
  --dataset_name mimic_iv_cdm \
  --mimic_iv_cdm_root_dir /data/EHR_data_public/mimic-iv-cdm \
  --mimic_iv_cdm_concept_map_dir /data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS \
  --output_dir .cache/ethos_vocab/mimic_iv_cdm/main_disease \
  --num_workers 0 \
  --num_processes 8 \
  --process_chunk_size 1000 \
  --checkpoint_path .cache/ethos_vocab/mimic_iv_cdm/main_disease/checkpoint.pkl \
  --checkpoint_every_chunks 5 \
  --overwrite_output
