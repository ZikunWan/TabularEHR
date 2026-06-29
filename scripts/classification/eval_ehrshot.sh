#!/bin/bash
set -e

TASKS=(
    "guo_los"
    "guo_readmission"
    "guo_icu"
    "lab_anemia"
    "lab_hyperkalemia"
    "lab_hyponatremia"
    "lab_hypoglycemia"
    "lab_thrombocytopenia"
    "new_acutemi"
    "new_celiac"
    "new_hyperlipidemia"
    "new_hypertension"
    "new_lupus"
    "new_pancan"
)

for TASK_IDX in "${!TASKS[@]}"; do
    TASK="${TASKS[$TASK_IDX]}"
    CUDA_VISIBLE_DEVICES=$((TASK_IDX % 8)) python test/classification/test_candidate_decoder.py \
        --dataset_name "ehrshot" \
        --data_dir "/data/EHR_data_public/EHRSHOT" \
        --split_info_path "/data/EHR_data_public/EHRSHOT/index/ehrshot_test.csv" \
        --embedding_cache "/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt" \
        --checkpoint_dir "/data/zikun_workspace/checkpoints/ehrshot/${TASK}/candidate_decoder" \
        --task_name "${TASK}" \
        --type_vocab_file "data/type_vocab.json" \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_candidate/ehrshot_task_candidate_embeddings.pt" \
        --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
        --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --max_table_len 4096 \
        --batch_size 32 \
        --max_eval_samples 1000 &

    if (( (TASK_IDX + 1) % 8 == 0 )); then
        wait
    fi
done

wait
