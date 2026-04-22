#!/bin/bash

python ./pretraining/bi_reconstruct_eval.py \
    --root_dir "/data/zikun_workspace/mimic-iv-3.1_tabular" \
    --test_sample_info_path "/data/zikun_workspace/mimic-iv-3.1_tabular/task_index/val/bi_reconstruct.csv" \
    --type_vocab_file "/data/zikun_workspace/code/data/type_vocab.json" \
    --table_text_embedding "/data/zikun_workspace/mimic-iv-3.1_tabular/embeddings/table_text_embeddings.pt" \
    --llm_path "/data/model_weights_public/BlueZeros/EHR-R1-1.7B" \
    --table_encoder_path "/data/zikun_workspace/checkpoints/pretraining/stage2_bi_reconstruct/tabular_encoder" \
    --task "text_to_table" \
    --random_subset_size 1000 \
    --max_samples 20 \
    --max_new_tokens 256 \
    --print_samples 5 \
    --output_path "/data/zikun_workspace/code/data/mimic/bi_reconstruct_text_to_table_predictions.jsonl"
