set -e

cd /data/zikun_workspace/code/train/Classifier

deepspeed --include localhost:4,5,6,7 train_mimic_iv_cdm_classifier.py \
    --deepspeed "/data/zikun_workspace/code/ds_config_zero2.json" \
    --embedding_cache "/data/zikun_workspace/.cache/embeddings/mimic_iv_cdm/text_embeddings_stage2.pt" \
    --output_dir "/data/zikun_workspace/checkpoints/mimic_iv_cdm/main_diagnosis/phenotype_triplet_learning" \
    --pretrained_path "/data/zikun_workspace/checkpoints/pretraining/phenotype_triplet_learning" \
    --run_name "mimic_iv_cdm_query_classifier_phenotype_triplet_learning" \
    --query_encoder knowledge \
    --query_embedding_cache "/data/zikun_workspace/.cache/embeddings/query_classifier/mimic_iv_cdm_task_query_knowledge_embeddings.pt" \
    --knowledge_encoder_path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
    --knowledge_encoder_base_model_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --max_table_len 16384 \
    --per_device_train_batch_size 16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5
