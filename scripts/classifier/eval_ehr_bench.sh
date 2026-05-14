#!/bin/bash
set -e
export MIMIC_SKIP_SAMPLE_CACHE_CHECK=1
cd "$(dirname "$0")/../../test/Classifier"

for task_name in \
    ED_Hospitalization \
    ED_Inpatient_Mortality \
    ED_ICU_Tranfer_12hour \
    ED_Reattendance_3day \
    ED_Critical_Outcomes \
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
    ICU_Readmission
do
    CUDA_VISIBLE_DEVICES=0 python test_ehr_bench_classifier.py \
        --data_dir /data/zikun_workspace/mimic-iv-3.1_tabular \
        --sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/test/${task_name}.csv" \
        --embedding_cache /data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings.pt \
        --checkpoint_dir "/data/zikun_workspace/checkpoints/ehr_bench/${task_name}/table_encoder/llm_query_scratch" \
        --task_name "$task_name" \
        --type_vocab_file /data/zikun_workspace/code/data/type_vocab.json \
        --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt \
        --query_llm_model_path /data/model_weights_public/BlueZeros/EHR-R1-1.7B \
        --max_table_len 16384 \
        --batch_size 64
done
