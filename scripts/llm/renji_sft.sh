HF_HOME=/home/ma-user/sfs_turbo/sai6/zkwan/.cache accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ./ds_config_zero2.json \
    --zero_stage 2 \
    /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/evaluation/renji_sft.py \
    --model_name_or_path "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/model_weights/BlueZeros/EHR-R1-1.7B" \
    --root_dir "./data/Renji" \
    --max_train_samples 200000 \
    --eval_strategy "no" \
    --max_seq_length 8192 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --bf16 True \
    --use_peft True \
    --lora_r 16 \
    --lora_alpha 32 \
    --report_to "wandb" \
    --dataset_type "map" \
    --gradient_checkpointing True \
    --target_metrics "CR,血糖,尿酸,甘油三脂,总胆固醇,血氨" \
    --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-MetabolicRenal" \
    --run_name "renji_EHR-R1-1.7B_MetabolicRenal" \
    --cache_dir "/home/ma-user/sfs_turbo/sai6/zkwan/.cache/renji_sft" \


python /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/renji_test.py \
    --model_path "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-MetabolicRenal" \
    --base_model_path "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/model_weights/BlueZeros/EHR-R1-1.7B" \
    --root_dir "./data/Renji" \
    --split "test" \
    --max_samples 50000 \
    --tp_size 4 \
    --target_metrics "CR,血糖,尿酸,甘油三脂,总胆固醇,血氨"

HF_HOME=/home/ma-user/sfs_turbo/sai6/zkwan/.cache accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file ./ds_config_zero2.json \
    --zero_stage 2 \
    /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/evaluation/renji_sft.py \
    --model_name_or_path "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/model_weights/BlueZeros/EHR-R1-1.7B" \
    --root_dir "./data/Renji" \
    --max_train_samples 200000 \
    --eval_strategy "no" \
    --max_seq_length 8192 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 5 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --bf16 True \
    --use_peft True \
    --lora_r 16 \
    --lora_alpha 32 \
    --report_to "wandb" \
    --dataset_type "map" \
    --gradient_checkpointing True \
    --target_metrics "CMV-DNA,EBV-DNA,HBsAg,HBsAb,HBeAg,HBeAb,HBcAb" \
    --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-VirusActivation" \
    --run_name "renji_EHR-R1-1.7B_VirusActivation" \
    --cache_dir "/home/ma-user/sfs_turbo/sai6/zkwan/.cache/renji_sft" \


python /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/renji_test.py \
    --model_path "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-VirusActivation" \
    --base_model_path "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/model_weights/BlueZeros/EHR-R1-1.7B" \
    --root_dir "./data/Renji" \
    --split "test" \
    --max_samples 50000 \
    --tp_size 4 \
    --target_metrics "CMV-DNA,EBV-DNA,HBsAg,HBsAb,HBeAg,HBeAb,HBcAb"
# Group 1:
# --target_metrics "ALT,AST,ALP,γ-GT,TB,DB,胆汁酸,TP,ALB,PT,INR" \
# --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-GraftInjury" \
# --run_name "renji_EHR-R1-1.7B_GraftInjury"

# Group 2:
# --target_metrics "他克莫司浓度,环孢素谷浓度,环孢素峰浓度,雷帕浓度" \
# --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-DrugConc" \
# --run_name "renji_EHR-R1-1.7B_DrugConc"

# Group 3:
# --target_metrics "WBC,N(%),淋巴细胞绝对值,嗜酸性粒细胞百分比,HB,PLT" \
# --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-ImmuneInfection" \
# --run_name "renji_EHR-R1-1.7B_ImmuneInfection"

# Group 4:
# --target_metrics "CR,血糖,尿酸,甘油三脂,总胆固醇,血氨" \
# --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-MetabolicRenal" \
# --run_name "renji_EHR-R1-1.7B_MetabolicRenal

# Group 5:
# --target_metrics "CMV-DNA,EBV-DNA,HBsAg,HBsAb,HBeAg,HBeAb,HBcAb" \
# --output_dir "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-VirusActivation" \
# --run_name "renji_EHR-R1-1.7B_VirusActivation"