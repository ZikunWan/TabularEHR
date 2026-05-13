#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python train/ethos/build_dataset_vocab.py \
  --dataset_name eicu \
  --eicu_root_dir /data/EHR_data_public/eicu-crd/2.0 \
  --eicu_processed_dir /data/zikun_workspace/eicu-crd/processed \
  --output_dir .cache/ethos_vocab/eicu \
  --num_workers 8 \
  --overwrite_output
