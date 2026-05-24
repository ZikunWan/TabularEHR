#!/bin/bash
set -e
export MIMIC_SKIP_SAMPLE_CACHE_CHECK=1
NUM_GPUS=$(nvidia-smi -L | wc -l)
TASK_INDEX_ROOT="/data/zikun_workspace/mimic-iv-3.1_tabular/task_index"

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

PRETRAIN_NAMES=(
    #"next_token_prediction"
    "contrastive_learning"
    #"task_query_classification"
)

PRETRAIN_PATHS=(
    #"/data/zikun_workspace/checkpoints/pretraining/next_token_prediction_mimic_eicu_ehrshot"
    "/data/zikun_workspace/checkpoints/pretraining/contrastive_learning_mimic_eicu_ehrshot"
    #"/data/zikun_workspace/checkpoints/pretraining/task_query_classification_mimic_eicu_ehrshot"
)

for i in "${!PRETRAIN_NAMES[@]}"; do
    PRETRAIN_NAME="${PRETRAIN_NAMES[$i]}"
    PRETRAIN_PATH="${PRETRAIN_PATHS[$i]}"

    for TASK in "${TASKS[@]}"; do
        echo "==================================="
        echo "Training EHR-Bench Task: $TASK"
        echo "Pretrained Path: $PRETRAIN_NAME"
        echo "==================================="

        TRAIN_INFO_PATH="${TASK_INDEX_ROOT}/train/${TASK}.csv"
        VAL_INFO_PATH="${TASK_INDEX_ROOT}/val/${TASK}.csv"
        OUTPUT_DIR="/data/zikun_workspace/checkpoints/ehr_bench/${TASK}/table_encoder/llm_query_${PRETRAIN_NAME}_mimic_eicu_ehrshot"

        cd /data/zikun_workspace/code/train/Classifier
        deepspeed --num_gpus=$NUM_GPUS train_ehr_bench_classifier.py \
            --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
            --output_dir "$OUTPUT_DIR" \
            --run_name "ehr_bench_${TASK}_llm_query_${PRETRAIN_NAME}" \
            --pretrained_path "$PRETRAIN_PATH" \
            --task_name "$TASK" \
            --train_sample_info_path "$TRAIN_INFO_PATH" \
            --val_sample_info_path "$VAL_INFO_PATH" \
            --max_table_len 16384 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 64 \
            --num_train_epochs 50 \
            --learning_rate 1e-5 \
            --early_stopping_patience 10 \
            --max_train_samples 3000 \
            --max_eval_samples 1000

        cd /data/zikun_workspace/code/test/Classifier
        CUDA_VISIBLE_DEVICES=0 python test_ehr_bench_classifier.py \
            --data_dir /data/zikun_workspace/mimic-iv-3.1_tabular \
            --sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/test/${TASK}.csv" \
            --embedding_cache /data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings.pt \
            --checkpoint_dir "$OUTPUT_DIR" \
            --task_name "$TASK" \
            --type_vocab_file /data/zikun_workspace/code/data/type_vocab.json \
            --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt \
            --query_llm_model_path /data/model_weights_public/BlueZeros/EHR-R1-1.7B \
            --max_table_len 16384 \
            --batch_size 64 \
            --pretrained_path "$PRETRAIN_PATH"
    done
done
