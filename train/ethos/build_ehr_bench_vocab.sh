#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python train/ethos/build_dataset_vocab.py \
  --dataset_name ehr_bench \
  --ehr_bench_data_dir /data/zikun_workspace/mimic-iv-3.1_tabular \
  --output_dir .cache/ethos_vocab/ehr_bench \
  --num_workers 8 \
  --overwrite_output
