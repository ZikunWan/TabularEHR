#!/bin/bash
set -e

NUM_GPUS=$(nvidia-smi -L | wc -l)
cd train/Classifier

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
    deepspeed --num_gpus=$NUM_GPUS train_eicu_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --output_dir "/data/zikun_workspace/checkpoints/eicu/phenotype_query_contrastive_learning/${TASK}" \
    --run_name "eicu_${TASK}_phenotype_query_contrastive_learning" \
    --task_name "$TASK" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/phenotype_query_contrastive_learning" \
    --max_table_len 16384 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 50 \
    --learning_rate 1e-5 \
    --max_train_samples 10000
done
