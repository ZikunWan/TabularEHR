#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTHONWARNINGS=",ignore:pkg_resources is deprecated as an API:UserWarning:mlflow.utils.requirements_utils"

cd /data/zikun_workspace/code/test/Llama

if [ ! -s /data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/meds_encoder/llama_base_4096_clmbr/classification_head.bin ]; then
  echo "Skipping evaluation because classification_head.bin was not found"
  exit 0
fi

python test_mimic_iv_cdm_llama.py \
  --checkpoint_dir /data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/meds_encoder/llama_base_4096_clmbr \
  --output_dir /data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/meds_encoder/llama_base_4096_clmbr \
  --root_dir /data/EHR_data_public/mimic-iv-cdm \
  --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
  --max_seq_length 4096 \
  --batch_size 64 \
  --concept_map_dir /data/EHR_data_public/mimic-iv-3.1-meds/pre_MEDS
