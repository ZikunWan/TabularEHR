#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "[1/3] Running next token prediction pretraining..."
bash scripts/pretraining/next_token_prediction.sh

echo "[2/3] Running task query classification pretraining..."
bash scripts/pretraining/task_query_classification.sh

echo "[3/3] Running contrastive learning pretraining..."
bash scripts/pretraining/contrastive_learning.sh

echo "All pretraining stages completed."
