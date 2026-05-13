#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python train/ethos/build_dataset_vocab.py \
  --dataset_name ehrshot \
  --ehrshot_root_dir /data/EHR_data_public/EHRSHOT \
  --output_dir .cache/ethos_vocab/ehrshot \
  --num_workers 8 \
  --overwrite_output
