#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

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
  python test/ethos/test_eicu_ethos.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/ethos/base" \
    --output_dir "/data/zikun_workspace/checkpoints/eicu/${TASK_NAME}/ethos/base/eval_logs" \
    --root_dir /data/EHR_data_public/eicu-crd/2.0 \
    --processed_dir /data/zikun_workspace/eicu-crd/processed \
    --sample_info_test_path /data/zikun_workspace/eicu-crd/processed/sample_info_test.json \
    --vocab_dir .cache/ethos_vocab/eicu \
    --task_name "$TASK_NAME" \
    --max_seq_length 4096 \
    --batch_size 16
done
