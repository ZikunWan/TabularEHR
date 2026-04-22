#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
TABLE_MODE="table_only"
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/CausalLM

for TASK_NAME in \
    "lab_hypoglycemia"
    #"new_acutemi" \
    #"new_celiac" \
    #"new_hyperlipidemia" \
    #"new_hypertension" \
    #"new_lupus" \
    #"new_pancan" \
    #"guo_los" \
    #"guo_readmission" \
    #"guo_icu" \
    #"lab_hyperkalemia" \
    #"lab_hyponatremia" \
    #"lab_anemia" \
    #"lab_thrombocytopenia" \
do
    TASK_KEY=$TASK_NAME

    for MODEL_KEY in \
        "qwen3_5_9b"
        #"gpt_oss_20b" 
        #"medgemma_1_5_4b_it" \
        #"ehr_r1_1_7b" \
        
        
    do
        if [ "$MODEL_KEY" = "gpt_oss_20b" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehrshot/${TASK_KEY}/${TABLE_MODE}/gpt_oss_20b"
        elif [ "$MODEL_KEY" = "medgemma_1_5_4b_it" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehrshot/${TASK_KEY}/${TABLE_MODE}/medgemma_1_5_4b_it"
        elif [ "$MODEL_KEY" = "qwen3_5_9b" ]; then
            MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehrshot/${TASK_KEY}/${TABLE_MODE}/qwen3_5_9b"
        else
            MODEL_PATH="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehrshot/${TASK_KEY}/${TABLE_MODE}/ehr_r1_1_7b"
        fi

        if [ ! -f "$MODEL_PATH/model.safetensors" ] && [ ! -f "$MODEL_PATH/classification_head.bin" ]; then
            echo "Skipping $MODEL_KEY because neither model.safetensors nor classification_head.bin was found in $MODEL_PATH"
            continue
        fi

        python test_ehrshot_llm.py \
            --model_path "$MODEL_PATH" \
            --output_dir "$MODEL_PATH" \
            --root_dir "/home/ma-user/sfs_turbo/sai6/zkwan/EHRSHOT" \
            --task_name "$TASK_NAME" \
            --table_mode "$TABLE_MODE" \
            --max_seq_len 32768 \
            --tp_size "$NUM_GPUS" \
            --use_sequence_classification True \
            --max_samples 1000
    done
done
