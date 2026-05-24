#!/bin/bash
set -euo pipefail

mkdir -p /data/zikun_workspace/code/.cache/meds_encoder_tokenizers/renji

cd /data/zikun_workspace/code/train/Llama

python build_dataset_tokenizer_config.py \
  --dataset_name renji \
  --model_name_or_path /data/model_weights_public/StanfordShahLab/llama-base-4096-clmbr \
  --output_tokenizer_config_path /data/zikun_workspace/code/.cache/meds_encoder_tokenizers/renji/expanded_tokenizer_config.json \
  --renji_root_dir /data/EHR_data_public/Renji \
  --renji_eval_split test \
  --overwrite_output true \
  --show_progress true
