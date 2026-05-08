#!/bin/bash
set -e

cd /data/zikun_workspace/code/train/Classifier

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

for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"

    if [ $((i % 2)) -eq 0 ]; then
        GPU_GROUP="0,1,2,3"
        MASTER_PORT=29500
    else
        GPU_GROUP="4,5,6,7"
        MASTER_PORT=29501
    fi

    deepspeed --include "localhost:${GPU_GROUP}" --master_port="$MASTER_PORT" train_eicu_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --data_dir "/data/EHR_data_public/eicu-crd/2.0" \
    --processed_dir "/data/zikun_workspace/eicu-crd/processed" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/eicu/text_embeddings_clinicalbert.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/eicu/${TASK}" \
    --run_name "eicu_${TASK}" \
    --task_name "$TASK" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification" \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_embeddings.pt" \
    --query_text_encoder_path "/data/zikun_workspace/checkpoints/pretraining/text_encoder_stage2/epoch_5.pt" \
    --query_text_encoder_base_model "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --max_table_len 32768 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 50 \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100 \
    --max_train_samples 10000 &

    if [ $((i % 2)) -eq 1 ]; then
        wait
    fi
done

wait
