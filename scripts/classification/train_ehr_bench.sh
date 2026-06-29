#!/bin/bash
set -e
source "$(dirname "$0")/../common/silent_info.sh"

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

for TASK in "${TASKS[@]}"; do
    MIMIC_SKIP_SAMPLE_CACHE_CHECK=1 deepspeed --include localhost:4,5,6,7 train/classification/train_candidate_decoder.py \
        --deepspeed "ds_config_zero2.json" \
        --dataset_name "ehr_bench" \
        --output_dir "/data/zikun_workspace/checkpoints/ehr_bench/${TASK}/table_encoder/candidate_decoder" \
        --run_name "ehr_bench_${TASK}_candidate_decoder" \
        --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/joint" \
        --fine_tune_mode "full_fine_tune" \
        --task_name "${TASK}" \
        --data_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
        --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt" \
        --train_sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/${TASK}.csv" \
        --val_sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val/${TASK}.csv" \
        --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_candidate/ehr_bench_task_candidate_embeddings.pt" \
        --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
        --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --max_table_len 16384 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 64 \
        --num_train_epochs 100 \
        --learning_rate 1e-5 \
        --lr_scheduler_type cosine_with_min_lr \
        --lr_scheduler_kwargs '{"min_lr": 1e-6}' \
        --early_stopping_patience 10 \
        --max_train_samples 3000 \
        --max_eval_samples 1000
done
