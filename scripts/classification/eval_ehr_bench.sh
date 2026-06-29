#!/bin/bash
set -e

TASKS=(
    "ED_Hospitalization"
    "ED_Inpatient_Mortality"
    "ED_ICU_Tranfer_12hour"
    "ED_Reattendance_3day"
    "ED_Critical_Outcomes"
    "Readmission_30day"
    "Readmission_60day"
    "Inpatient_Mortality"
    "LengthOfStay_3day"
    "LengthOfStay_7day"
    "ICU_Mortality_1day"
    "ICU_Mortality_2day"
    "ICU_Mortality_3day"
    "ICU_Mortality_7day"
    "ICU_Mortality_14day"
    "ICU_Stay_7day"
    "ICU_Stay_14day"
    "ICU_Readmission"
)

for TASK_IDX in "${!TASKS[@]}"; do
    TASK="${TASKS[$TASK_IDX]}"
    MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 CUDA_VISIBLE_DEVICES=$((TASK_IDX % 8)) python test/classification/test_candidate_decoder.py \
        --dataset_name "ehr_bench" \
        --data_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
        --sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/test/${TASK}.csv" \
        --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt" \
        --checkpoint_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK}/table_encoder/candidate_decoder" \
        --task_name "${TASK}" \
        --type_vocab_file "data/type_vocab.json" \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_candidate/ehr_bench_task_candidate_embeddings.pt" \
        --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
        --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --max_table_len 16384 \
        --batch_size 64 \
        --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/joint" &

    if (( (TASK_IDX + 1) % 8 == 0 )); then
        wait
    fi
done

wait
