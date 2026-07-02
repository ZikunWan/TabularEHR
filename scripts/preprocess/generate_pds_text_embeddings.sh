#!/bin/bash
set -e

NPROC="${NPROC:-8}"

python preprocess/pds/generate_text_embeddings.py \
    --stage harvest \
    --root-dir "/data/zikun_workspace/input/tables/PDS" \
    --cache-dir "/data/zikun_workspace/.cache/embeddings/PDS"

torchrun --nproc_per_node="${NPROC}" preprocess/pds/generate_text_embeddings.py \
    --stage encode \
    --root-dir "/data/zikun_workspace/input/tables/PDS" \
    --cache-dir "/data/zikun_workspace/.cache/embeddings/PDS" \
    --model-path "/data/zikun_workspace/checkpoints/pretraining/knowledge_encoder/clinicalBERT_after_stage2/best.pt" \
    --base-model-path "/data/model_weights_public/emilyalsentzer/Bio_ClinicalBERT" \
    --final-output "/data/zikun_workspace/.cache/embeddings/PDS/text_embeddings_stage2.pt"
