#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
export TOKENIZERS_PARALLELISM=false

TABLE_MODE="table_only"
PROJECT_NAME="eicu_llm"
export WANDB_PROJECT=$PROJECT_NAME

cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/train/CausalLM

for TASK_NAME in \
    "mortality" \
    "long_term_mortality" \
    "readmission" \
    "los_3day" \
    "los_7day" \
    "creatinine" \
    "bilirubin" \
    "platelets" \
    "wbc" \
    "diagnosis" \
    "final_acuity" \
    "imminent_discharge"
do
    TASK_KEY=$TASK_NAME

    for MODEL_NAME in \
        "qwen3_5_9b"\
        "gpt-oss-20b" \
        "medgemma-1.5-4b-it" \
        "ehr_r1_1_7b"
    do
        if [ "$MODEL_NAME" = "gpt-oss-20b" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/model_weights/gpt-oss-20b"
            MODEL_KEY="gpt_oss_20b"
            BATCH_SIZE=2
        elif [ "$MODEL_NAME" = "medgemma-1.5-4b-it" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/model_weights/google/medgemma-1.5-4b-it"
            MODEL_KEY="medgemma_1_5_4b_it"
            BATCH_SIZE=4
        elif [ "$MODEL_NAME" = "qwen3_5_9b" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/model_weights/Qwen/Qwen3.5-9B"
            MODEL_KEY="qwen3_5_9b"
            BATCH_SIZE=4
        else
            MODEL_PATH="/home/ma-user/sfs_turbo/model_weights/EHR-R1-1.7B"
            MODEL_KEY="ehr_r1_1_7b"
            BATCH_SIZE=4
        fi

        RUN_NAME="eicu_${TASK_KEY}_${TABLE_MODE}_${MODEL_KEY}_finetune"
        OUTPUT_DIR="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/eicu/${TASK_KEY}/${TABLE_MODE}/${MODEL_KEY}"

        export WANDB_NAME=$RUN_NAME

        accelerate launch --num_processes=$NUM_GPUS train_eicu_llm.py \
            --deepspeed "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/ds_config_zero2.json" \
            --model_name_or_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --run_name "$RUN_NAME" \
            --report_to wandb \
            --task_name "$TASK_NAME" \
            --use_sequence_classification True \
            --max_seq_len 32768 \
            --per_device_train_batch_size "$BATCH_SIZE" \
            --gradient_accumulation_steps 1 \
            --num_train_epochs 1 \
            --learning_rate 2e-5 \
            --logging_steps 100 \
            --save_steps 500 \
            --save_total_limit 3 \
            --save_strategy "steps" \
            --bf16 True \
            --gradient_checkpointing True \
            --dataloader_num_workers 8 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --table_mode "$TABLE_MODE" \
            --max_train_samples 10000
    done
done
