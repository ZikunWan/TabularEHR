#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

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
  python test/ethos/test_ehrshot_ethos.py \
    --checkpoint_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK_NAME}/ethos/base" \
    --output_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK_NAME}/ethos/base/eval_logs" \
    --root_dir /data/EHR_data_public/EHRSHOT \
    --test_info_path /data/EHR_data_public/EHRSHOT/index/ehrshot_test.csv \
    --vocab_dir .cache/ethos_vocab/ehrshot \
    --task_name "$TASK_NAME" \
    --max_seq_length 4096 \
    --batch_size 32 \
    --max_samples 1000
done
