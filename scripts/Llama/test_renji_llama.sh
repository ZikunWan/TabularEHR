#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTHONWARNINGS="${PYTHONWARNINGS:-},ignore:pkg_resources is deprecated as an API:UserWarning:mlflow.utils.requirements_utils"

cd /data/zikun_workspace/code/test/Llama

CHECKPOINT_DIR="/data/zikun_workspace/checkpoints/renji/meds_encoder/llama_base_4096_clmbr"
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_logs"

if [ ! -s "${CHECKPOINT_DIR}/classification_head.bin" ]; then
  echo "Skipping Renji Llama eval: classification_head.bin not found"
  exit 0
fi

python test_renji_llama.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --root_dir /data/EHR_data_public/Renji \
  --split test \
  --target_prediction_points day0,day30,day180,day365 \
  --max_seq_length 4096 \
  --batch_size 16
