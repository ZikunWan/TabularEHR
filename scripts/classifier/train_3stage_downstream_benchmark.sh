#!/bin/bash
set -e

export MIMIC_SKIP_SAMPLE_CACHE_CHECK=1

ROOT_DIR="/data/zikun_workspace/code"
CHECKPOINT_ROOT="/data/zikun_workspace/checkpoints"
CACHE_ROOT="/data/zikun_workspace/.cache/embeddings"
NUM_GPUS=$(nvidia-smi -L | wc -l)
TASK_INDEX_ROOT="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index"

PRETRAIN_NAMES=(
    "next_token_prediction"
    "task_query_classification"
    "phenotype_metric_learning"
)

PRETRAIN_PATHS=(
    "${CHECKPOINT_ROOT}/pretraining/next_token_prediction"
    "${CHECKPOINT_ROOT}/pretraining/task_query_classification"
    "${CHECKPOINT_ROOT}/pretraining/phenotype_metric_learning"
)

QUERY_ENCODERS=(
    "llm"
    "llm"
    "knowledge"
)

DATASETS=(
    "EHR-Bench"
    "eICU"
    "EHRSHOT"
)

EICU_TASKS=(
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

EHR_BENCH_TASKS=(
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

EHRSHOT_TASKS=(
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

log_stage() {
    echo "==================================="
    echo "$1"
    echo "==================================="
}

skip_if_result_exists() {
    local result_path="$1"
    local label="$2"

    if [ -f "${result_path}" ]; then
        log_stage "Skipping ${label} because result already exists: ${result_path}"
        return 0
    fi

    return 1
}

run_mimic_iv_cdm() {
    local pretrain_name="$1"
    local pretrain_path="$2"
    local output_dir="${CHECKPOINT_ROOT}/mimic_iv_cdm/main_diagnosis/${pretrain_name}"
    local result_file="${output_dir}/test_results_metrics.csv"

    if skip_if_result_exists "${result_file}" "MIMIC-IV-CDM | ${pretrain_name}"; then
        return 0
    fi

    log_stage "MIMIC-IV-CDM | ${pretrain_name}"

    cd "${ROOT_DIR}/train/Classifier"
    deepspeed --num_gpus="${NUM_GPUS}" train_mimic_iv_cdm_classifier.py \
        --deepspeed "${ROOT_DIR}/ds_config_zero2.json" \
        --embedding_cache "${CACHE_ROOT}/mimic_iv_cdm/text_embeddings_stage2.pt" \
        --output_dir "${output_dir}" \
        --pretrained_path "${pretrain_path}" \
        --run_name "mimic_iv_cdm_main_diagnosis_${pretrain_name}" \
        --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
        --query_encoder llm \
        --query_embedding_cache "${CACHE_ROOT}/query_classifier/mimic_iv_cdm_task_query_llm_embeddings.pt" \
        --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
        --max_table_len 16384 \
        --per_device_train_batch_size 16 \
        --num_train_epochs 100 \
        --learning_rate 1e-5

    cd "${ROOT_DIR}/test/Classifier"
    CUDA_VISIBLE_DEVICES=0 python test_mimic_iv_cdm_classifier.py \
        --data_dir "/data/EHR_data_public/mimic-iv-cdm" \
        --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
        --checkpoint_dir "${output_dir}" \
        --task_name "MIMIC-IV-CDM Main Disease Diagnoses" \
        --type_vocab_file "${ROOT_DIR}/data/type_vocab.json" \
        --query_encoder llm \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/mimic_iv_cdm_task_query_llm_embeddings.pt" \
        --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
        --max_table_len 16384
}

run_renji() {
    local pretrain_name="$1"
    local pretrain_path="$2"
    local output_dir="${CHECKPOINT_ROOT}/renji/${pretrain_name}"
    local result_file="${output_dir}/test_results_test_auroc.csv"

    if skip_if_result_exists "${result_file}" "Renji | ${pretrain_name}"; then
        return 0
    fi

    log_stage "Renji | ${pretrain_name}"

    cd "${ROOT_DIR}/train/Classifier"
    deepspeed --num_gpus="${NUM_GPUS}" train_renji_classifier.py \
        --deepspeed "${ROOT_DIR}/ds_config_zero2.json" \
        --output_dir "${output_dir}" \
        --run_name "renji_query_classifier_${pretrain_name}" \
        --max_table_len 16384 \
        --per_device_train_batch_size 32 \
        --num_train_epochs 50 \
        --learning_rate 1e-5 \
        --query_encoder llm \
        --query_embedding_cache "${CACHE_ROOT}/query_classifier/renji_task_query_llm_embeddings.pt" \
        --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
        --pretrained_path "${pretrain_path}"

    cd "${ROOT_DIR}/test/Classifier"
    CUDA_VISIBLE_DEVICES=0 python test_renji_classifier.py \
        --data_dir "/data/EHR_data_public/Renji" \
        --embedding_cache "/data/zikun_workspace/.cache/embeddings/renji/text_embeddings_stage2.pt" \
        --checkpoint_dir "${output_dir}" \
        --split "test" \
        --type_vocab_file "${ROOT_DIR}/data/type_vocab.json" \
        --query_encoder llm \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/renji_task_query_llm_embeddings.pt" \
        --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
        --max_table_len 16384
}

run_eicu() {
    local pretrain_name="$1"
    local pretrain_path="$2"
    local query_encoder="$3"
    local query_embedding_cache="${CACHE_ROOT}/query_classifier/eicu_task_query_${query_encoder}_embeddings.pt"

    for task_name in "${EICU_TASKS[@]}"; do
        local output_dir="${CHECKPOINT_ROOT}/eicu/${pretrain_name}/${task_name}"
        local result_file="${output_dir}/test_results_metrics.csv"

        if skip_if_result_exists "${result_file}" "eICU | ${pretrain_name} | ${task_name}"; then
            continue
        fi

        log_stage "eICU | ${pretrain_name} | ${task_name}"

        cd "${ROOT_DIR}/train/Classifier"
        deepspeed --num_gpus="${NUM_GPUS}" train_eicu_classifier.py \
            --deepspeed "${ROOT_DIR}/ds_config_zero2.json" \
            --output_dir "${output_dir}" \
            --run_name "eicu_${task_name}_${pretrain_name}" \
            --task_name "${task_name}" \
            --pretrained_path "${pretrain_path}" \
            --embedding_cache "${CACHE_ROOT}/eicu/text_embeddings_stage2.pt" \
            --query_encoder "${query_encoder}" \
            --query_embedding_cache "${query_embedding_cache}" \
            --max_table_len 16384 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 32 \
            --num_train_epochs 50 \
            --learning_rate 1e-5 \
            --max_train_samples 10000

        cd "${ROOT_DIR}/test/Classifier"
        CUDA_VISIBLE_DEVICES=0 python test_eicu_classifier.py \
            --data_dir "/data/EHR_data_public/eicu-crd/2.0" \
            --processed_dir "/data/zikun_workspace/eicu-crd/processed" \
            --sample_info_val_path "/data/zikun_workspace/eicu-crd/processed/sample_info_val.json" \
            --sample_info_test_path "/data/zikun_workspace/eicu-crd/processed/sample_info_test.json" \
            --embedding_cache "/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt" \
            --checkpoint_dir "${output_dir}" \
            --task_name "${task_name}" \
            --type_vocab_file "${ROOT_DIR}/data/type_vocab.json" \
            --query_encoder "${query_encoder}" \
            --query_embedding_cache "${query_embedding_cache}" \
            --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
            --max_table_len 16384 \
            --batch_size 32
    done
}

run_ehr_bench() {
    local pretrain_name="$1"
    local pretrain_path="$2"
    local query_encoder="$3"
    local query_embedding_cache="${CACHE_ROOT}/query_classifier/ehr_bench_task_query_${query_encoder}_embeddings.pt"

    for task_name in "${EHR_BENCH_TASKS[@]}"; do
        local train_info_path="${TASK_INDEX_ROOT}/train/${task_name}.csv"
        local val_info_path="${TASK_INDEX_ROOT}/val/${task_name}.csv"
        local test_info_path="${TASK_INDEX_ROOT}/test/${task_name}.csv"
        local output_dir="${CHECKPOINT_ROOT}/ehr_bench/${pretrain_name}/${task_name}"
        local result_file="${output_dir}/test_results_metrics.csv"

        if skip_if_result_exists "${result_file}" "EHR-Bench | ${pretrain_name} | ${task_name}"; then
            continue
        fi

        log_stage "EHR-Bench | ${pretrain_name} | ${task_name}"

        cd "${ROOT_DIR}/train/Classifier"
        deepspeed --num_gpus="${NUM_GPUS}" train_ehr_bench_classifier.py \
            --deepspeed "${ROOT_DIR}/ds_config_zero2.json" \
            --output_dir "${output_dir}" \
            --run_name "ehr_bench_${task_name}_${query_encoder}_query_${pretrain_name}" \
            --pretrained_path "${pretrain_path}" \
            --task_name "${task_name}" \
            --train_sample_info_path "${train_info_path}" \
            --val_sample_info_path "${val_info_path}" \
            --query_encoder "${query_encoder}" \
            --query_embedding_cache "${query_embedding_cache}" \
            --max_table_len 16384 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 64 \
            --num_train_epochs 50 \
            --learning_rate 1e-5 \
            --early_stopping_patience 10 \
            --max_train_samples 3000 \
            --max_eval_samples 1000

        cd "${ROOT_DIR}/test/Classifier"
        CUDA_VISIBLE_DEVICES=0 python test_ehr_bench_classifier.py \
            --data_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
            --sample_info_path "${test_info_path}" \
            --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt" \
            --checkpoint_dir "${output_dir}" \
            --task_name "${task_name}" \
            --type_vocab_file "${ROOT_DIR}/data/type_vocab.json" \
            --query_encoder "${query_encoder}" \
            --query_embedding_cache "${query_embedding_cache}" \
            --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
            --max_table_len 16384 \
            --batch_size 64 \
            --pretrained_path "${pretrain_path}"
    done
}

run_ehrshot() {
    local pretrain_name="$1"
    local pretrain_path="$2"
    local query_encoder="$3"
    local query_embedding_cache="${CACHE_ROOT}/query_classifier/ehrshot_task_query_${query_encoder}_embeddings.pt"
    local output_root="${CHECKPOINT_ROOT}/ehrshot/${pretrain_name}"

    for task_name in "${EHRSHOT_TASKS[@]}"; do
        local checkpoint_dir="${output_root}/${task_name}"
        local result_file="${checkpoint_dir}/test_results_metrics.csv"

        if skip_if_result_exists "${result_file}" "EHRSHOT | ${pretrain_name} | ${task_name}"; then
            continue
        fi

        log_stage "EHRSHOT | ${pretrain_name} | ${task_name}"

        cd "${ROOT_DIR}/train/Classifier"
        deepspeed --num_gpus="${NUM_GPUS}" train_ehrshot_classifier.py \
            --deepspeed "${ROOT_DIR}/ds_config_zero2.json" \
            --output_dir "${output_root}" \
            --run_name "ehrshot_${task_name}_${pretrain_name}" \
            --task_name "${task_name}" \
            --pretrained_path "${pretrain_path}" \
            --query_encoder "${query_encoder}" \
            --query_embedding_cache "${query_embedding_cache}" \
            --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
            --max_table_len 8192 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 64 \
            --num_train_epochs 50 \
            --learning_rate 1e-5 \
            --max_train_samples 500 \
            --max_eval_samples 1000 \
            --early_stopping_patience 10

        cd "${ROOT_DIR}/test/Classifier"
        CUDA_VISIBLE_DEVICES=0 python test_ehrshot_classifier.py \
            --data_dir "/data/EHR_data_public/EHRSHOT" \
            --split_info_path "/data/EHR_data_public/EHRSHOT/index/ehrshot_test.csv" \
            --embedding_cache "/data/zikun_workspace/.cache/embeddings/ehrshot/text_embeddings_stage2.pt" \
            --checkpoint_dir "${checkpoint_dir}" \
            --task_name "${task_name}" \
            --type_vocab_file "${ROOT_DIR}/data/type_vocab.json" \
            --query_encoder "${query_encoder}" \
            --query_embedding_cache "${query_embedding_cache}" \
            --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
            --max_table_len 8192 \
            --batch_size 32 \
            --max_eval_samples 1000
    done
}

for dataset_name in "${DATASETS[@]}"; do
    log_stage "Starting downstream benchmark for ${dataset_name}"

    for i in "${!PRETRAIN_NAMES[@]}"; do
        pretrain_name="${PRETRAIN_NAMES[$i]}"
        pretrain_path="${PRETRAIN_PATHS[$i]}"
        query_encoder="${QUERY_ENCODERS[$i]}"

        case "${dataset_name}" in
            "MIMIC-IV-CDM")
                run_mimic_iv_cdm "${pretrain_name}" "${pretrain_path}"
                ;;
            "Renji")
                run_renji "${pretrain_name}" "${pretrain_path}"
                ;;
            "eICU")
                run_eicu "${pretrain_name}" "${pretrain_path}" "${query_encoder}"
                ;;
            "EHR-Bench")
                run_ehr_bench "${pretrain_name}" "${pretrain_path}" "${query_encoder}"
                ;;
            "EHRSHOT")
                run_ehrshot "${pretrain_name}" "${pretrain_path}" "${query_encoder}"
                ;;
        esac
    done
done
