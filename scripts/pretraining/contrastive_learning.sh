export GPUS_PER_NODE=4

torchrun --nproc_per_node=$GPUS_PER_NODE ./pretraining/contrastive_learning.py \
    --deepspeed "./ds_config_zero2.json" \
    --root_dir "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular" \
    --train_info_path "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/train/contrastive_learning.csv" \
    --val_info_path "/home/ma-user/sfs_turbo/sai6/zkwan/mimic-iv-3.1_tabular/task_index/val/contrastive_learning.csv" \
    --num_negatives 1024 \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 16 \
    --learning_rate 1e-4 \
    --temperature 0.07 \
    --warmup_steps 50 \
    --num_train_epochs 100 \
    --logging_steps 10  \
    --save_steps 50 \
    --run_name "0.2_ratio" \
    --run_project "contrastive_learning" \
    --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/contrastive_learning" \
    --sort_by_table_length True \
    --short_table_ratio 0.2 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --early_stopping_patience 20 \
    --metric_for_best_model "eval_recall@10"
