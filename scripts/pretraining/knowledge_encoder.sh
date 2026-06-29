#!/bin/bash
source "$(dirname "$0")/../common/silent_info.sh"

if [ "$CACHE_ONLY" = "true" ]; then
    python ./pretraining/knowledge_encoder.py \
        --concept_path "/data/zikun_workspace/knowledge/CONCEPT.csv" \
        --concept_relationship_path "/data/zikun_workspace/knowledge/CONCEPT_RELATIONSHIP.csv" \
        --triple_cache "/data/zikun_workspace/.cache/pretraining/triples_cache" \
        --kg_max_triples "None" \
        --output_dir "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2" \
        --cache_only
else
    deepspeed --num_gpus=8 ./pretraining/knowledge_encoder.py \
        --deepspeed "./ds_config_zero2.json" \
        --model_name_or_path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
        --concept_path "/data/zikun_workspace/knowledge/CONCEPT.csv" \
        --concept_relationship_path "/data/zikun_workspace/knowledge/CONCEPT_RELATIONSHIP.csv" \
        --triple_cache "/data/zikun_workspace/.cache/pretraining/triples_cache" \
        --kg_max_triples "None" \
        --kg_num_negatives 4 \
        --kg_margin 1.0 \
        --kg_distance_p 1 \
        --kg_relation_reg 1e-4 \
        --max_length 128 \
        --batch_size 256 \
        --epochs 5 \
        --learning_rate 2e-5 \
        --min_lr 1e-6 \
        --weight_decay 0.01 \
        --warmup_ratio 0.05 \
        --num_workers 8 \
        --bf16 \
        --logging_steps 50 \
        --save_steps 100 \
        --save_total_limit 1 \
        --report_to "wandb" \
        --wandb_project "knowledge_encoder" \
        --wandb_run_name "stage2_concept_relationship_transe_zero2" \
        --output_dir "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2" 
fi
