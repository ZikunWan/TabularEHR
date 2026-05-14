#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python train/ethos/build_dataset_vocab.py \
  --dataset_name ehrshot \
  --ehrshot_root_dir /data/EHR_data_public/EHRSHOT \
  --output_dir .cache/ethos_vocab/ehrshot \
  --num_workers 0 \
  --num_processes 8 \
  --process_chunk_size 1000 \
  --checkpoint_path .cache/ethos_vocab/ehrshot/checkpoint.pkl \
  --checkpoint_every_chunks 5 \
  --overwrite_output
