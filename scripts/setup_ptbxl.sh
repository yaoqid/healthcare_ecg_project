#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_SLUG="${1:-khyeh0719/ptb-xl-dataset}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/data/ptb-xl}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ptbxl_download.XXXXXX")"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is required but was not found in PATH." >&2
    exit 1
  fi
}

check_kaggle_auth() {
  if [[ -f "${HOME}/.kaggle/access_token" ]]; then
    return
  fi

  if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
    return
  fi

  if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    return
  fi

  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    return
  fi

  echo "Error: Kaggle credentials were not found." >&2
  echo "Set up one of the following before running this script:" >&2
  echo "  1. ~/.kaggle/access_token" >&2
  echo "  2. KAGGLE_API_TOKEN environment variable" >&2
  echo "  3. ~/.kaggle/kaggle.json (legacy)" >&2
  echo "  4. KAGGLE_USERNAME and KAGGLE_KEY environment variables (legacy)" >&2
  echo "You may also need to accept the dataset terms once on Kaggle in your browser." >&2
  exit 1
}

resolve_dataset_root() {
  local search_root="$1"
  local metadata_path

  metadata_path="$(find "${search_root}" -type f -name "ptbxl_database.csv" -print -quit)"
  if [[ -z "${metadata_path}" ]]; then
    return 1
  fi

  dirname "${metadata_path}"
}

require_command kaggle
check_kaggle_auth

if [[ -f "${OUTPUT_DIR}/ptbxl_database.csv" && -f "${OUTPUT_DIR}/scp_statements.csv" ]]; then
  echo "PTB-XL already appears to be set up at:"
  echo "  ${OUTPUT_DIR}"
  echo "Skipping download."
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"

echo "Downloading PTB-XL from Kaggle dataset '${DATASET_SLUG}'..."
kaggle datasets download "${DATASET_SLUG}" -p "${TMP_DIR}" --unzip

DATASET_ROOT="$(resolve_dataset_root "${TMP_DIR}")" || {
  echo "Error: Could not find ptbxl_database.csv after download." >&2
  exit 1
}

if [[ ! -f "${DATASET_ROOT}/scp_statements.csv" ]]; then
  echo "Error: Downloaded PTB-XL data is missing scp_statements.csv." >&2
  exit 1
fi

if [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Error: Output directory is not empty:" >&2
  echo "  ${OUTPUT_DIR}" >&2
  echo "Move or remove existing files, then rerun the script." >&2
  exit 1
fi

shopt -s dotglob nullglob
for item in "${DATASET_ROOT}"/*; do
  mv "${item}" "${OUTPUT_DIR}/"
done
shopt -u dotglob nullglob

if [[ ! -f "${OUTPUT_DIR}/ptbxl_database.csv" || ! -f "${OUTPUT_DIR}/scp_statements.csv" ]]; then
  echo "Error: PTB-XL setup completed, but required metadata files are still missing." >&2
  exit 1
fi

echo
echo "PTB-XL is ready at:"
echo "  ${OUTPUT_DIR}"
echo
echo "Detected contents:"
[[ -d "${OUTPUT_DIR}/records100" ]] && echo "  - records100/"
[[ -d "${OUTPUT_DIR}/records500" ]] && echo "  - records500/"
echo "  - ptbxl_database.csv"
echo "  - scp_statements.csv"
echo
echo "Next steps:"
echo "  bash scripts/train_ptbxl_models.sh"
