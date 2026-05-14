#!/bin/bash
set -e

cd "$(dirname "$0")/../../test/Classifier"

for task_name in \
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
    CUDA_VISIBLE_DEVICES=0 python test_eicu_classifier.py \
        --data_dir /data/EHR_data_public/eicu-crd/2.0 \
        --processed_dir /data/zikun_workspace/eicu-crd/processed \
        --sample_info_val_path /data/zikun_workspace/eicu-crd/processed/sample_info_val.json \
        --sample_info_test_path /data/zikun_workspace/eicu-crd/processed/sample_info_test.json \
        --embedding_cache /data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt \
        --checkpoint_dir "/data/zikun_workspace/checkpoints/eicu/${task_name}" \
        --task_name "$task_name" \
        --type_vocab_file /data/zikun_workspace/code/data/type_vocab.json \
        --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt \
        --query_llm_model_path /data/model_weights_public/BlueZeros/EHR-R1-1.7B \
        --max_table_len 16384 \
        --batch_size 32
done
