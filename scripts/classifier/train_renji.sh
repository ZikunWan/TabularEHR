#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd train/Classifier

deepspeed --num_gpus=$NUM_GPUS train_renji_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --output_dir "/data/zikun_workspace/checkpoints/renji/scratch" \
    --run_name "renji_query_classifier_llm_adapter_scratch_full" \
    --max_table_len 16384 \
    --per_device_train_batch_size 32 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
#    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification" \

