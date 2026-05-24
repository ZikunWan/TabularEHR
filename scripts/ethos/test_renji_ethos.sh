#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

python test/ethos/test_renji_ethos.py \
  --checkpoint_dir /data/zikun_workspace/checkpoints/renji/ethos/base \
  --output_dir /data/zikun_workspace/checkpoints/renji/ethos/base/eval_logs \
  --root_dir /data/EHR_data_public/Renji \
  --vocab_dir .cache/ethos_vocab/renji \
  --task_name multi_label_prediction \
  --split test \
  --max_seq_length 4096 \
  --batch_size 64
