#!/bin/bash
set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)
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

for TASK in "${TASKS[@]}"; do
    deepspeed --num_gpus=$NUM_GPUS train/classification/train_eicu_classifier.py \
    --deepspeed "ds_config_zero2.json" \
    --output_dir "/data/zikun_workspace/checkpoints/eicu/phenotype_query_contrastive_learning/${TASK}" \
    --run_name "eicu_${TASK}_phenotype_query_contrastive_learning" \
    --task_name "$TASK" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/phenotype_query_contrastive_learning" \
    --query_encoder llm \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/eicu_task_query_llm_embeddings.pt" \
    --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
    --max_table_len 16384 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 50 \
    --learning_rate 1e-5 \
    --max_train_samples 10000
done
