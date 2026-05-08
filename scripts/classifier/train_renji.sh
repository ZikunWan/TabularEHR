#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/train/Classifier

torchrun --nproc_per_node=$NUM_GPUS train_renji_classifier.py \
    --deepspeed "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/ds_config_zero2.json" \
    --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/renji_classifier_1d_full_with_0.3_ratio" \
    --run_name "full_0.3_ratio_pretraining" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification" \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_embeddings.pt" \
    --query_text_encoder_path "/data/zikun_workspace/checkpoints/pretraining/text_encoder_stage2/epoch_5.pt" \
    --query_text_encoder_base_model "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --max_table_len 4096 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 25 \
    --learning_rate 2e-4 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 100
