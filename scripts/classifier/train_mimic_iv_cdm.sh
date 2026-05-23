set -e

cd /data/zikun_workspace/code/train/Classifier

deepspeed --include localhost:0,1,2,3 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/query_classifier_llm_adapter_task_query_classification_3dataset_full" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/task_query_classification_mimic_eicu_ehrshot" \
    --run_name "mimic_iv_cdm_main_diagnosis_query_classifier_llm_adapter_task_query_classification_3dataset_full" \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/task_query_llm_embeddings.pt" \
    --query_llm_model_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
    --max_table_len 16384 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 &

