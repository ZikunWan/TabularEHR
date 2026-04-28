#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTHONWARNINGS="${PYTHONWARNINGS:-},ignore:pkg_resources is deprecated as an API:UserWarning:mlflow.utils.requirements_utils"

cd /data/zikun_workspace/code/test/Llama

has_eval_result() {
  [ -s "$1/metrics.csv" ] && [ -s "$1/raw_predictions.csv" ] && return 0
  [ -s "$1/eval_logs/metrics.csv" ] && [ -s "$1/eval_logs/raw_predictions.csv" ] && return 0
  return 1
}

for TASK_NAME in \
  guo_los \
  guo_readmission \
  guo_icu \
  lab_anemia \
  lab_hyperkalemia \
  lab_hyponatremia \
  lab_hypoglycemia \
  lab_thrombocytopenia \
  new_acutemi \
  new_celiac \
  new_hyperlipidemia \
  new_hypertension \
  new_lupus \
  new_pancan
do
  if [ ! -s "/data/zikun_workspace/checkpoints/ehrshot/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr/classification_head.bin" ]; then
    echo "Skipping ${TASK_NAME}: classification_head.bin not found"
    continue
  fi
  if has_eval_result "/data/zikun_workspace/checkpoints/ehrshot/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr"; then
    echo "[SKIP] Existing eval result found for ${TASK_NAME}"
    continue
  fi
  python test_ehrshot_llama.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr" \
    --output_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr" \
    --task_name "$TASK_NAME" \
    --max_seq_length 4096 \
    --batch_size 32 \
    --max_test_samples 1000
done
