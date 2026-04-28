#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

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
  python test/ethos/test_ehr_bench_ethos.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/ethos/base" \
    --output_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK_NAME}/ethos/base/eval_logs" \
    --data_dir /data/zikun_workspace/mimic-iv-3.1_tabular \
    --vocab_dir .cache/ethos_vocab/ehr_bench \
    --task_name "$TASK_NAME" \
    --itemid_representation code \
    --max_seq_length 4096 \
    --batch_size 64
done
