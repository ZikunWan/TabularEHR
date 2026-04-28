#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTHONWARNINGS=",ignore:pkg_resources is deprecated as an API:UserWarning:mlflow.utils.requirements_utils"

cd /data/zikun_workspace/code/test/Llama

has_eval_result() {
  [ -s "$1/metrics.csv" ] && [ -s "$1/raw_predictions.csv" ] && return 0
  [ -s "$1/eval_logs/metrics.csv" ] && [ -s "$1/eval_logs/raw_predictions.csv" ] && return 0
  return 1
}

for TASK_NAME in \
  Readmission_30day \
  Readmission_60day \
  Inpatient_Mortality \
  LengthOfStay_3day \
  LengthOfStay_7day \
  ICU_Mortality_1day \
  ICU_Mortality_2day \
  ICU_Mortality_3day \
  ICU_Mortality_7day \
  ICU_Mortality_14day \
  ICU_Stay_7day \
  ICU_Stay_14day \
  ICU_Readmission \
  ED_Hospitalization \
  ED_Inpatient_Mortality \
  ED_ICU_Tranfer_12hour \
  ED_Reattendance_3day \
  ED_Critical_Outcomes
do
  if [ ! -s "/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr/classification_head.bin" ]; then
    echo "Skipping ${TASK_NAME}: classification_head.bin not found"
    continue
  fi
  if has_eval_result "/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr"; then
    echo "[OVERWRITE] Existing eval result found for ${TASK_NAME}, rerunning"
  fi
  python test_ehr_bench_llama.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr" \
    --output_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/meds_encoder/llama_base_4096_clmbr" \
    --data_dir /data/zikun_workspace/mimic-iv-3.1_tabular \
    --task_name "$TASK_NAME" \
    --itemid_representation code \
    --max_seq_length 4096 \
    --batch_size 64 \
    --concept_map_dir ""
done
