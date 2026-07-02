#!/bin/bash
set -e
source "$(dirname "$0")/../common/silent_info.sh"

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

GPU_INCLUDE="localhost:0,1,2,3,4,5,6,7"
DATA_DIR="/data/zikun_workspace/input/tables/PDS"
PATIENT_SPLIT_PATH="/data/zikun_workspace/input/tasks/classification/PDS/indices/patient_splits.json"
EMBEDDING_CACHE="/data/zikun_workspace/.cache/embeddings/PDS/text_embeddings_stage2.pt"
QUERY_CACHE="/data/zikun_workspace/.cache/embeddings/query_candidate/PDS_task_candidate_embeddings.pt"
CHECKPOINT_ROOT="/data/zikun_workspace/checkpoints/PDS"
FORCE_RETRAIN="0"
MAX_TRAIN_PATIENTS="500"
MAX_EVAL_PATIENTS="300"

if [[ ! -f "${EMBEDDING_CACHE}" ]]; then
    echo "Missing PDS embedding cache: ${EMBEDDING_CACHE}" >&2
    exit 1
fi

if [[ ! -f "${PATIENT_SPLIT_PATH}" ]]; then
    echo "Missing PDS patient split file: ${PATIENT_SPLIT_PATH}" >&2
    exit 1
fi

has_completed_checkpoint() {
    local output_dir="$1"
    local checkpoint
    local step

    for checkpoint in "${output_dir}"/checkpoint-*; do
        [[ -d "${checkpoint}" ]] || continue
        [[ -f "${checkpoint}/trainer_state.json" ]] || continue

        step="${checkpoint##*-}"
        if compgen -G "${checkpoint}/global_step${step}/*model_states.pt" >/dev/null; then
            return 0
        fi
        if compgen -G "${checkpoint}"'/*.safetensors' >/dev/null; then
            return 0
        fi
        if compgen -G "${checkpoint}"'/*.bin' >/dev/null; then
            return 0
        fi
    done

    return 1
}

for TRIAL in "${TRIALS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        if [[ "${TRIAL}" == "149" && "${TASK}" == "adverse_event_next_visit" ]]; then
            continue
        fi

        OUTPUT_DIR="${CHECKPOINT_ROOT}/${TRIAL}/${TASK}/candidate_decoder"
        if [[ "${FORCE_RETRAIN}" != "1" ]] && has_completed_checkpoint "${OUTPUT_DIR}"; then
            echo "Skipping completed PDS task: trial=${TRIAL}, task=${TASK}, output_dir=${OUTPUT_DIR}"
            continue
        fi

        EXTRA_ARGS=()
        if [[ -n "${MAX_TRAIN_PATIENTS}" ]]; then
            EXTRA_ARGS+=(--max_train_patients "${MAX_TRAIN_PATIENTS}")
        fi
        if [[ -n "${MAX_EVAL_PATIENTS}" ]]; then
            EXTRA_ARGS+=(--max_eval_patients "${MAX_EVAL_PATIENTS}")
        fi

        deepspeed --include "${GPU_INCLUDE}" train/classification/train_candidate_decoder.py \
            --deepspeed "ds_config_zero2.json" \
            --dataset_name "pds" \
            --output_dir "${OUTPUT_DIR}" \
            --run_name "PDS_${TRIAL}_${TASK}_candidate_decoder" \
            --task_name "${TASK}" \
            --trial_id "${TRIAL}" \
            --data_dir "${DATA_DIR}" \
            --patient_split_path "${PATIENT_SPLIT_PATH}" \
            --embedding_cache "${EMBEDDING_CACHE}" \
            --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/joint" \
            --fine_tune_mode "full_fine_tune" \
            --query_embedding_cache "${QUERY_CACHE}" \
            --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
            --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
            --max_table_len 4096 \
            --per_device_train_batch_size 128 \
            --per_device_eval_batch_size 64 \
            --num_train_epochs 50 \
            --learning_rate 1e-5 \
            --lr_scheduler_type cosine_with_min_lr \
            --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
            --early_stopping_patience 10 \
            "${EXTRA_ARGS[@]}"
    done
done
