#!/bin/bash
cd /home/ma-user/modelarts/user-job-dir/LiverTransplantation/tabular/test/Classifier

TASKS=(
    "ED_Hospitalization"
    "ED_Inpatient_Mortality"
    "ED_ICU_Tranfer_12hour"
    "ED_Reattendance_3day"
    "ED_Critical_Outcomes"
    "Readmission_30day"
    "Readmission_60day"
    "Inpatient_Mortality"
    "LengthOfStay_3day"
    "LengthOfStay_7day"
    "ICU_Mortality_1day"
    "ICU_Mortality_2day"
    "ICU_Mortality_3day"
    "ICU_Mortality_7day"
    "ICU_Mortality_14day"
    "ICU_Stay_7day"
    "ICU_Stay_14day"
    "ICU_Readmission"
)

BASE_CHECKPOINT_DIR="/home/ma-user/sfs_turbo/sai6/zkwan/checkpoints/ehr_bench"
CHECKPOINT_SUFFIX="_using_stage1_pretraining"
AGGREGATE_FILE="${BASE_CHECKPOINT_DIR}/all_metrics${CHECKPOINT_SUFFIX}.csv"

echo "task,auroc,auprc,accuracy,n_samples" > "$AGGREGATE_FILE"

for TASK in "${TASKS[@]}"; do
    echo "==================================="
    echo "Evaluating EHR-Bench Task: $TASK"
    echo "==================================="

    TASK_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/${TASK}${CHECKPOINT_SUFFIX}"
    if [ ! -d "$TASK_CHECKPOINT_DIR" ]; then
        echo "Warning: Checkpoint directory $TASK_CHECKPOINT_DIR does not exist. Skipping."
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 python test_ehr_bench_classifier.py \
        --checkpoint_dir "$TASK_CHECKPOINT_DIR" \
        --task_name "$TASK" \
        --batch_size 64

    METRICS_FILE="${TASK_CHECKPOINT_DIR}/test_results_metrics.csv"
    if [ -f "$METRICS_FILE" ]; then
        tail -n +2 "$METRICS_FILE" >> "$AGGREGATE_FILE"
    else
        echo "Warning: Metrics file not found: $METRICS_FILE"
    fi
done

echo "==================================="
echo "All done! Aggregate metrics saved to $AGGREGATE_FILE"
cat "$AGGREGATE_FILE" | column -s, -t
