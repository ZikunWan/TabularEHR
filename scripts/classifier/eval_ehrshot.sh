#!/bin/bash
set -e

cd "$(dirname "$0")/../../test/Classifier"

for task_name in \
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
    CUDA_VISIBLE_DEVICES=0 python test_ehrshot_classifier.py \
        --data_dir /data/EHR_data_public/EHRSHOT \
        --split_info_path /data/EHR_data_public/EHRSHOT/index/ehrshot_test.csv \
        --embedding_cache /data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt \
        --checkpoint_dir "/data/zikun_workspace/checkpoints/ehrshot/classifier/${task_name}" \
        --task_name "$task_name" \
        --type_vocab_file /data/zikun_workspace/code/data/type_vocab.json \
        --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt \
        --query_llm_model_path /data/model_weights_public/BlueZeros/EHR-R1-1.7B \
        --max_table_len 8192 \
        --batch_size 32 \
        --max_eval_samples 1000
done
