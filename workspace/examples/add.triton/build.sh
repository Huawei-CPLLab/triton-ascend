#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../../scripts/path.sh"

OUTPUT_DIR="${SCRIPT_DIR}/${WS_OUTPUT_DIR_NAME}"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export WS_KERNEL_OUTPUT_DIR="${OUTPUT_DIR}"

TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python3 "${SCRIPT_DIR}/Add.py" "${OUTPUT_DIR}"
