#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PTBXL_DIR="${1:-${PROJECT_ROOT}/ptb-xl}"
SAMPLING_RATE="${2:-100}"
SEQ_LEN="${3:-3}"

if [[ ! -f "${PTBXL_DIR}/ptbxl_database.csv" || ! -f "${PTBXL_DIR}/scp_statements.csv" ]]; then
  echo "Error: PTB-XL dataset was not found at:" >&2
  echo "  ${PTBXL_DIR}" >&2
  echo "Run 'bash scripts/setup_ptbxl.sh' first or pass the correct PTB-XL folder." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "Training CNN on PTB-XL..."
python train_cnn.py --real-data --ptbxl-dir "${PTBXL_DIR}" --sampling-rate "${SAMPLING_RATE}"

echo
echo "Training LSTM on PTB-XL..."
python train_lstm.py \
  --real-data \
  --ptbxl-dir "${PTBXL_DIR}" \
  --sampling-rate "${SAMPLING_RATE}" \
  --seq-len "${SEQ_LEN}" \
  --selection-metric balanced_acc \
  --threshold-metric balanced_acc
