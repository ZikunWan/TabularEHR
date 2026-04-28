#!/bin/bash
set -euo pipefail

cd /data/zikun_workspace/code

python test/ethos/test_mimic_iv_cdm_ethos.py \
  --checkpoint_dir /data/zikun_workspace/checkpoints/mimic_iv_cdm/main_disease/ethos/base \
  --output_dir /data/zikun_workspace/checkpoints/mimic_iv_cdm/main_disease/ethos/base/eval_logs \
  --root_dir /data/EHR_data_public/mimic-iv-cdm \
  --concept_map_dir /data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS \
  --vocab_dir .cache/ethos_vocab/mimic_iv_cdm/main_disease \
  --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
  --max_seq_length 4096 \
  --batch_size 64
