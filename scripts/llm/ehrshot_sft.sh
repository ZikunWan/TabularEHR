#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
export TOKENIZERS_PARALLELISM=false

TABLE_MODE="table_only"
PROJECT_NAME="ehrshot_llm"
export WANDB_PROJECT=$PROJECT_NAME

cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/train/CausalLM

for TASK_NAME in \
    "guo_readmission" \
    "guo_icu" \
    "lab_hyperkalemia" \
    "lab_hypoglycemia" \
    "new_acutemi" \
    "new_celiac" \
    "new_hyperlipidemia" \
    "new_hypertension" \
    "new_lupus" \
    "new_pancan"
    #"lab_thrombocytopenia" \
    #"lab_hyponatremia" \
    #"lab_anemia" \
    #"guo_los" \
do
    TASK_KEY=$TASK_NAME

    for MODEL_NAME in \
        "qwen3_5_9b"\
        "ehr_r1_1_7b" \
        "medgemma_1_5_4b_it" \
        "gpt_oss_20b"
    do
        if [ "$MODEL_NAME" = "gpt_oss_20b" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/model_weights/gpt-oss-20b"
            MODEL_KEY="gpt_oss_20b"
            BATCH_SIZE=1
        elif [ "$MODEL_NAME" = "medgemma_1_5_4b_it" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/model_weights/google/medgemma-1.5-4b-it"
            MODEL_KEY="medgemma_1_5_4b_it"
            BATCH_SIZE=5
        elif [ "$MODEL_NAME" = "qwen3_5_9b" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/model_weights/Qwen/Qwen3.5-9B"
            MODEL_KEY="qwen3_5_9b"
            BATCH_SIZE=5
        else
            MODEL_PATH="/home/ma-user/sfs_turbo/model_weights/EHR-R1-1.7B"
            MODEL_KEY="ehr_r1_1_7b"
            BATCH_SIZE=5
        fi

        RUN_NAME="ehrshot_${TASK_KEY}_${TABLE_MODE}_${MODEL_KEY}_finetune"
        OUTPUT_DIR="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehrshot/${TASK_KEY}/${TABLE_MODE}/${MODEL_KEY}"

        export WANDB_NAME=$RUN_NAME

        accelerate launch --num_processes=$NUM_GPUS train_ehrshot_llm.py \
            --deepspeed "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/ds_config_zero3.json" \
            --root_dir "/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT" \
            --model_name_or_path "$MODEL_PATH" \
            --output_dir "$OUTPUT_DIR" \
            --run_name "$RUN_NAME" \
            --report_to wandb \
            --task_name "$TASK_NAME" \
            --use_sequence_classification True \
            --max_seq_len 8192 \
            --per_device_train_batch_size "$BATCH_SIZE" \
            --gradient_accumulation_steps 1 \
            --num_train_epochs 1 \
            --learning_rate 2e-5 \
            --logging_steps 100 \
            --save_steps 500 \
            --save_total_limit 3 \
            --save_strategy "steps" \
            --bf16 True \
            --gradient_checkpointing False \
            --dataloader_num_workers 8 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --table_mode "$TABLE_MODE" \
            --max_train_samples 500 \
            --max_eval_samples 1000
            
    done
done
