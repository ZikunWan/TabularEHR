#!/bin/bash
set -e

TRIALS=(
    "102"
    "103"
    "105"
    "118"
    "119"
    "121"
    "122"
    "127"
    "128"
    "149"
)

TASKS=(
    "severe_outcome"
    "adverse_event_next_visit"
)

DATA_DIR="/data/zikun_workspace/input/tables/PDS"
PATIENT_SPLIT_PATH="/data/zikun_workspace/input/tasks/classification/PDS/indices/patient_splits.json"
EMBEDDING_CACHE="/data/zikun_workspace/.cache/embeddings/PDS/text_embeddings_stage2.pt"
QUERY_CACHE="/data/zikun_workspace/.cache/embeddings/query_candidate/PDS_task_candidate_embeddings.pt"
CHECKPOINT_ROOT="/data/zikun_workspace/checkpoints/PDS"

if [[ ! -f "${EMBEDDING_CACHE}" ]]; then
    echo "Missing PDS embedding cache: ${EMBEDDING_CACHE}" >&2
    exit 1
fi

if [[ ! -f "${PATIENT_SPLIT_PATH}" ]]; then
    echo "Missing PDS patient split file: ${PATIENT_SPLIT_PATH}" >&2
    exit 1
fi

JOB_IDX=0
for TRIAL in "${TRIALS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        if [[ "${TRIAL}" == "149" && "${TASK}" == "adverse_event_next_visit" ]]; then
            continue
        fi

        CUDA_VISIBLE_DEVICES=$((JOB_IDX % 8)) python test/classification/test_candidate_decoder.py \
            --dataset_name "pds" \
            --data_dir "${DATA_DIR}" \
            --checkpoint_dir "${CHECKPOINT_ROOT}/${TRIAL}/${TASK}/candidate_decoder" \
            --task_name "${TASK}" \
            --trial_id "${TRIAL}" \
            --split "test" \
            --patient_split_path "${PATIENT_SPLIT_PATH}" \
            --embedding_cache "${EMBEDDING_CACHE}" \
            --type_vocab_file "data/type_vocab.json" \
            --query_embedding_cache "${QUERY_CACHE}" \
            --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
            --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
            --max_table_len 4096 \
            --batch_size 64 &

        JOB_IDX=$((JOB_IDX + 1))
        if (( JOB_IDX % 8 == 0 )); then
            wait
        fi
    done
done

wait
