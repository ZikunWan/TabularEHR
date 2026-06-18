#!/bin/bash
set -euo pipefail

SPEC_PATH="/data/zikun_workspace/.cache/phenotype_triplet_learning/phenotype_query_specs.json"
INPUT_DIR="/data/zikun_workspace/.cache/phenotype_triplet_learning/inputs"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/structEHR/bin/python}"

"${PYTHON_BIN}" ./pretraining/build_reference_phenotype_specs.py \
    --reference_path "./data/phenotype_triplet_reference_scales.csv" \
    --output_path "${SPEC_PATH}" \
    --statistic latest \
    --window_name full

MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 "${PYTHON_BIN}" ./pretraining/build_phenotype_metric_samples.py \
    --dataset mimic_iv eicu ehrshot \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/next_token_prediction.csv" \
    --val_sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val/next_token_prediction.csv" \
    --table_text_embedding "/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt" \
    --eicu_root_dir "/data/zikun_workspace/eicu-crd" \
    --eicu_processed_dir "/data/zikun_workspace/eicu-crd/processed" \
    --eicu_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_train.json" \
    --eicu_val_sample_info_path "/data/zikun_workspace/eicu-crd/processed/pretraining_index/sample_info_val.json" \
    --eicu_table_text_embedding "/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt" \
    --ehrshot_root_dir "/data/EHR_data_public/EHRSHOT" \
    --ehrshot_sample_info_path "/data/EHR_data_public/EHRSHOT/pretraining_index/sample_info_train.csv" \
    --ehrshot_val_sample_info_path "/data/EHR_data_public/EHRSHOT/pretraining_index/sample_info_val.csv" \
    --ehrshot_table_text_embedding "/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt" \
    --phenotype_spec_path "${SPEC_PATH}" \
    --output_dir "${INPUT_DIR}" \
    --splits train val \
    --num_workers 128 \
    --progress_update_interval 32 \
    --overwrite_manifest true \
    --min_table_rows 2
