#!/bin/bash
set -e
export MIMIC_SKIP_SAMPLE_CACHE_CHECK=1
NUM_GPUS=$(nvidia-smi -L | wc -l)

TASKS=(
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
)

for task_idx in "${!TASKS[@]}"; do
    task_name="${TASKS[$task_idx]}"
    gpu_id=$((task_idx % NUM_GPUS))

    echo "Evaluating ${task_name} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="$gpu_id" python test/classification/test_ehr_bench_classifier.py \
        --data_dir /data/zikun_workspace/mimic-iv-3.1_tabular \
        --sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/test/${task_name}.csv" \
        --embedding_cache /data/zikun_workspace/.cache/embeddings/mimic_iv/text_embeddings_stage2.pt \
        --checkpoint_dir "/data/zikun_workspace/checkpoints/ehr_bench/${task_name}/table_encoder/after_phenotype_query_contrastive_learning" \
        --task_name "$task_name" \
        --type_vocab_file data/type_vocab.json \
        --query_encoder llm \
        --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_classifier/ehr_bench_task_query_llm_embeddings.pt \
        --query_llm_model_path /data/model_weights_public/BlueZeros/EHR-R1-1.7B \
        --max_table_len 16384 \
        --batch_size 64 \
        --pretrained_path /data/zikun_workspace/checkpoints/pretraining/phenotype_query_contrastive_learning &

    if (( (task_idx + 1) % NUM_GPUS == 0 )); then
        wait
    fi
done

wait
