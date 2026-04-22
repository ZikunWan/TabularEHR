#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/train/Classifier

torchrun --nproc_per_node=$NUM_GPUS train_renji_classifier.py \
    --deepspeed "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/ds_config_zero2.json" \
    --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/renji_classifier_1d_full_with_0.3_ratio" \
    --run_name "full_0.3_ratio_pretraining" \
    --pretrained_path "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/stage1_contrastive_learning-v6_with_0.3_ratio/model.safetensors" \
    --per_device_train_batch_size 16 \
    --num_train_epochs 25 \
    --learning_rate 2e-4
