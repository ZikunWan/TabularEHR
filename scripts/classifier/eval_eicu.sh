#!/bin/bash
set -e

cd "$(dirname "$0")/../../test/Classifier"

NUM_GPUS=$(nvidia-smi -L | wc -l)
STAGES=(
    phenotype_query_contrastive_learning
)
TASKS=(
    mortality
    long_term_mortality
    readmission
    los_3day
    los_7day
    creatinine
    bilirubin
    platelets
    wbc
    final_acuity
    imminent_discharge
)

run_task() {
    local gpu_id="$1"
    local stage_name="$2"
    local task_name="$3"

    echo "Evaluating ${stage_name} | ${task_name} on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python test_eicu_classifier.py \
        --data_dir /data/EHR_data_public/eicu-crd/2.0 \
        --processed_dir /data/zikun_workspace/eicu-crd/processed \
        --sample_info_test_path /data/zikun_workspace/eicu-crd/processed/sample_info_test.json \
        --embedding_cache /data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_stage2.pt \
        --checkpoint_dir "/data/zikun_workspace/checkpoints/eicu/${stage_name}/${task_name}" \
        --task_name "${task_name}" \
        --type_vocab_file /data/zikun_workspace/code/data/type_vocab.json \
        --query_encoder llm \
        --query_embedding_cache /data/zikun_workspace/.cache/embeddings/query_classifier/eicu_task_query_llm_embeddings.pt \
        --query_llm_model_path /data/model_weights_public/BlueZeros/EHR-R1-1.7B \
        --max_table_len 16384 \
        --batch_size 32
}

for stage_name in "${STAGES[@]}"; do
    for i in "${!TASKS[@]}"; do
        gpu_id=$((i % NUM_GPUS))
        run_task "${gpu_id}" "${stage_name}" "${TASKS[$i]}" &

        if [ $(((i + 1) % NUM_GPUS)) -eq 0 ]; then
            wait
        fi
    done
    wait
done

wait
