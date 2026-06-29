#!/bin/bash
set -e

TASKS=(
    "mortality"
    "long_term_mortality"
    "readmission"
    "los_3day"
    "los_7day"
    "creatinine"
    "bilirubin"
    "platelets"
    "wbc"
    "final_acuity"
    "imminent_discharge"
)

for TASK_IDX in "${!TASKS[@]}"; do
    TASK="${TASKS[$TASK_IDX]}"
    CUDA_VISIBLE_DEVICES=$((TASK_IDX % 8)) python test/classification/test_candidate_decoder.py \
        --dataset_name "eicu" \
        --data_dir "/data/EHR_data_public/eicu-crd/2.0" \
        --processed_dir "/data/zikun_workspace/eicu-crd/processed" \
        --sample_info_test_path "/data/zikun_workspace/eicu-crd/processed/sample_info_test.json" \
        --embedding_cache "/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt" \
        --checkpoint_dir "/data/zikun_workspace/checkpoints/eicu/candidate_decoder/${TASK}" \
        --task_name "${TASK}" \
        --type_vocab_file "data/type_vocab.json" \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_candidate/eicu_task_candidate_embeddings.pt" \
        --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
        --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --max_table_len 16384 \
        --batch_size 32 &

    if (( (TASK_IDX + 1) % 8 == 0 )); then
        wait
    fi
done

wait
