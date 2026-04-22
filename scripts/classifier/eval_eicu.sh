#!/bin/bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/Classifier

TASKS=(
    "mortality"
    "long_term_mortality"
    "readmission"
    "los_3day"
    "los_7day"
    "creatinine"
    "bilirubin"
    "platelets"
    "wbc"
    "diagnosis"
    "final_acuity"
    "imminent_discharge"
)

# Run evaluation on CPU to avoid distributed issues with tests
export CUDA_VISIBLE_DEVICES=0

# Ensure we have a place to aggregate metrics
AGGREGATE_FILE="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/eicu_classifier/all_metrics.csv"
echo "task,auroc,accuracy,n_samples" > "$AGGREGATE_FILE"

for TASK in "${TASKS[@]}"; do
    echo "==================================="
    echo "Evaluating Task: $TASK"
    echo "==================================="
    
    CHECKPOINT_DIR="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/eicu_classifier/finetune_${TASK}"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo "Warning: Checkpoint directory $CHECKPOINT_DIR does not exist. Skipping."
        continue
    fi
    
    python test_eicu_classifier.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --task_name "$TASK" \
        --batch_size 16
        
    # Append the metrics to the aggregate file if successful
    METRICS_FILE="${CHECKPOINT_DIR}/test_results_metrics.csv"
    if [ -f "$METRICS_FILE" ]; then
        # Skip header and append
        tail -n +2 "$METRICS_FILE" >> "$AGGREGATE_FILE"
    fi
done

echo "==================================="
echo "All done! Aggregate metrics saved to $AGGREGATE_FILE"
cat "$AGGREGATE_FILE" | column -s, -t
