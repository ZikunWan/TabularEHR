#!/bin/bash
# Evaluation script for Renji dataset
# Works for both LoRA and full fine-tuning checkpoints.
# test_renji.py auto-detects LoRA by checking for adapter_config.json.

CHECKPOINT_DIR="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/renji_classifier_1d_full_with_0.3_ratio"
SPLIT="test"
BATCH_SIZE=32
TASK_MODE="multi_task"
SEED=42

#TARGET_METRICS="ALT,AST,TB"
#TARGET_POINTS="day14,day30"

echo "================================================="
echo "Evaluating checkpoint: $CHECKPOINT_DIR"
echo "split=$SPLIT"
echo "================================================="

python test_renji.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --batch_size $BATCH_SIZE \
    --task_mode $TASK_MODE \
    --split $SPLIT \
    --seed $SEED \
    ${TARGET_METRICS:+--target_metrics $TARGET_METRICS} \
    ${TARGET_POINTS:+--target_prediction_points $TARGET_POINTS}

echo "Done. Results saved to: $CHECKPOINT_DIR"
