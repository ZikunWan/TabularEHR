#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code/test/Llama

has_eval_result() {
  [ -s "$1/metrics.csv" ] && [ -s "$1/raw_predictions.csv" ] && return 0
  [ -s "$1/eval_logs/metrics.csv" ] && [ -s "$1/eval_logs/raw_predictions.csv" ] && return 0
  return 1
}

for TASK_NAME in \
  mortality \
  long_term_mortality \
  readmission \
  los_3day \
  los_7day \
  creatinine \
  bilirubin \
  platelets \
  wbc \
  final_acuity \
  imminent_discharge
do
  if [ ! -s "/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr/classification_head.bin" ]; then
    echo "[SKIP] classification_head.bin not found for ${TASK_NAME}"
    continue
  fi
  if has_eval_result "/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr"; then
    echo "[OVERWRITE] Existing eval result found, rerunning: ${TASK_NAME}"
  fi
  python test_eicu_llama.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr" \
    --task_name "$TASK_NAME" \
    --max_seq_length 4096 \
    --batch_size 16 \
    --output_dir "/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr/eval_logs" \
    --root_dir /data/EHR_data_public/eicu-crd/2.0 \
    --processed_dir /data/zikun_workspace/eicu-crd/processed
done
