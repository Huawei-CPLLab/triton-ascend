#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../../scripts/path.sh"

OUTPUT_DIR="${SCRIPT_DIR}/${WS_OUTPUT_DIR_NAME}"
if [ "$#" -eq 0 ]; then
    rm -rf "${OUTPUT_DIR}"
fi

make -C "${SCRIPT_DIR}" \
    ASCEND_HOME="${WS_CANN_PATH}" \
    TOOL_BIN="${WS_BISHENG_INSTALL_PATH}" \
    BISHENG_CC1="${WS_BISHENG_CC1_PATH}/bisheng" \
    OUT_DIR="${OUTPUT_DIR}" \
    "$@"
