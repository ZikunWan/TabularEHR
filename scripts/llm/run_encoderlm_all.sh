#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
  "eicu_encoderlm_test.sh"
  "ehrshot_encoderlm_sft.sh"
  "ehr_bench_encoderlm_sft.sh"
  "ehrshot_encoderlm_test.sh"
  "ehr_bench_encoderlm_test.sh"
)

echo "==> Running EncoderLM pipeline scripts in order"
echo "    Base dir: ${SCRIPT_DIR}"
echo

for script_name in "${SCRIPTS[@]}"; do
  script_path="${SCRIPT_DIR}/${script_name}"
  if [[ ! -f "${script_path}" ]]; then
    echo "[ERROR] Missing script: ${script_path}" >&2
    exit 1
  fi

  echo "----------------------------------------------------------------------"
  echo "[START] ${script_name}  ($(date '+%Y-%m-%d %H:%M:%S'))"
  bash "${script_path}"
  echo "[DONE ] ${script_name}  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo
done

echo "==> All scripts completed successfully."
