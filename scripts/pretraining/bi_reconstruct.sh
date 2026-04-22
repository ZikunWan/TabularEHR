export MIMIC_TABLE_LENGTH_WORKERS=64
export MIMIC_TABLE_LENGTH_CHUNK_SIZE=64

deepspeed --num_gpus=8 ./pretraining/bi_reconstruct.py \
    --deepspeed "./ds_config_zero2.json" \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --train_sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/train/bi_reconstruct.csv" \
    --type_vocab_file "/data/zikun_workspace/code/data/type_vocab.json" \
    --table_text_embedding "/data/zikun_workspace/mimic-iv-3.1_tabular/embeddings/table_text_embeddings.pt" \
    --llm_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
    --table_encoder_path "/data/zikun_workspace/checkpoints/contrastive_learning/tabular_encoder/model.safetensors" \
    --freeze_llm True \
    --freeze_table_encoder False \
    --attention_mode "1d" \
    --projector_hidden_size 2048 \
    --mask_ratio 0.15 \
    --max_masked_cells 64 \
    --random_subset_size 2000000 \
    --max_target_length 4096 \
    --per_device_train_batch_size 1 \
    --max_train_samples 1000000 \
    --gradient_accumulation_steps 32 \
    --dataloader_num_workers 16 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps "100" \
    --save_encoder_only True \
    --run_project "bi_reconstruct" \
    --run_name "stage2_bi_reconstruct" \
    --output_dir "/data/zikun_workspace/checkpoints/pretraining/stage2_bi_reconstruct"
