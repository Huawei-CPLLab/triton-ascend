#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../path.sh"

if [ ! -d "${WS_NPUIR_CLONE_PATH}/.git" ]; then
    "${WS_SCRIPTS_DIR}/clone/clone.npuir.sh"
fi

cd "${WS_NPUIR_CLONE_PATH}"

mkdir -p "${WS_BUILD_DIR_NAME}"

./build-tools/build.sh \
    --c-compiler clang --cxx-compiler clang++ \
    '--add-cmake-options=-DLLVM_ENABLE_LLD=ON' \
    --build-type Release \
    --enable-assertion \
    --disable-werror --disable-mlir-werror --disable-bishengir-werror \
    --build-triton \
    -t --bisheng-compiler "${WS_BISHENG_INSTALL_PATH}" \
    --collect-binary="${WS_NPUIR_CLONE_PATH}" \
    --build "${WS_BUILD_DIR_NAME}"

for tool in bishengir-translate; do
    if [ -f "${WS_NPUIR_CLONE_PATH}/${WS_BUILD_DIR_NAME}/install/bin/${tool}" ]; then
        cp -f "${WS_NPUIR_CLONE_PATH}/${WS_BUILD_DIR_NAME}/install/bin/${tool}" "${WS_NPUIR_CLONE_PATH}/bin/"
    elif [ -f "${WS_NPUIR_CLONE_PATH}/${WS_BUILD_DIR_NAME}/bin/${tool}" ]; then
        cp -f "${WS_NPUIR_CLONE_PATH}/${WS_BUILD_DIR_NAME}/bin/${tool}" "${WS_NPUIR_CLONE_PATH}/bin/"
    fi
done
