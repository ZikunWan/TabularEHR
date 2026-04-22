python /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/renji_test.py \
    --model_path "/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/EHR-R1-1.7B-Renji-text-DrugConc" \
    --base_model_path "/home/ma-user/modelarts/user-job-dir/LiverTransplantation/model_weights/BlueZeros/EHR-R1-1.7B" \
    --root_dir "./data/Renji" \
    --split "test" \
    --max_samples 50000 \
    --tp_size 4 \
    --target_metrics "他克莫司浓度,环孢素谷浓度,环孢素峰浓度,雷帕浓度,胆汁酸,γ-GT"
