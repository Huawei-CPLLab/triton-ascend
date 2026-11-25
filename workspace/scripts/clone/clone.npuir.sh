#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../path.sh"

mkdir -p "${WS_REPOS_PATH}"

echo "=== cloning npuir ==="

if [ ! -d "${WS_NPUIR_CLONE_PATH}/.git" ]; then
    git clone -b afshin/fix-kirin9020 https://gitcode.com/Fields-Compiler-Lab/AscendNPU-IR-Dev.git "${WS_NPUIR_CLONE_PATH}"
    cd "${WS_NPUIR_CLONE_PATH}"
    git submodule sync
    git submodule update --init --progress third-party/llvm-project
else
    echo "NPUIR already exists at ${WS_NPUIR_CLONE_PATH}. Skipping cloning."
fi
