#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/../path.sh"

compute_llvm_stamp() {
    local triton_dir="$1"
    local llvm_hash
    llvm_hash="$(tr -d '[:space:]' < "${triton_dir}/cmake/llvm-hash.txt")"
    local patch_dir="${triton_dir}/third_party/ascend/patch"
    local patch_hash="000000000000"
    if [ -d "${patch_dir}" ] && compgen -G "${patch_dir}/llvm_patch_*.patch" > /dev/null; then
        patch_hash="$(
            LC_ALL=C printf '%s\n' "${patch_dir}"/llvm_patch_*.patch \
                | LC_ALL=C sort \
                | xargs cat \
                | sha256sum \
                | cut -c1-12
        )"
    fi
    echo "${llvm_hash}-${patch_hash}"
}

if [ ! -d "${WS_TRITON_CLONE_PATH}/.git" ]; then
    "${WS_SCRIPTS_DIR}/clone/clone.triton.sh"
fi

EXPECTED_STAMP="$(compute_llvm_stamp "${WS_TRITON_CLONE_PATH}")"
INSTALL_STAMP_FILE="${WS_TRITON_LLVM_INSTALL_PATH}/.triton_llvm_stamp"

if [ -f "${INSTALL_STAMP_FILE}" ] \
    && [ "$(cat "${INSTALL_STAMP_FILE}")" = "${EXPECTED_STAMP}" ] \
    && [ -f "${WS_TRITON_LLVM_INSTALL_PATH}/lib/cmake/llvm/LLVMConfig.cmake" ] \
    && [ -x "${WS_TRITON_LLVM_INSTALL_PATH}/bin/FileCheck" ]; then
    echo "=== LLVM (${EXPECTED_STAMP}) is up to date; skipping LLVM build ==="
else
    echo "=== building LLVM (${EXPECTED_STAMP}) for triton-ascend ==="
    "${WS_SCRIPTS_DIR}/clone/clone.triton.sh"

    mkdir -p "${WS_TRITON_LLVM_CLONE_PATH}/${WS_BUILD_DIR_NAME}"
    cmake -S "${WS_TRITON_LLVM_CLONE_PATH}/llvm" \
        -B "${WS_TRITON_LLVM_CLONE_PATH}/${WS_BUILD_DIR_NAME}" \
        -G Ninja \
        -DCMAKE_C_COMPILER="${WS_CC}" \
        -DCMAKE_CXX_COMPILER="${WS_CXX}" \
        -DCMAKE_LINKER="${WS_LINKER}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_CCACHE_BUILD=ON \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_LLD=ON \
        -DCMAKE_INSTALL_PREFIX="${WS_TRITON_LLVM_INSTALL_PATH}"
    ninja -C "${WS_TRITON_LLVM_CLONE_PATH}/${WS_BUILD_DIR_NAME}" install
    cp -f "${WS_TRITON_LLVM_CLONE_PATH}/${WS_BUILD_DIR_NAME}/bin/FileCheck" \
        "${WS_TRITON_LLVM_INSTALL_PATH}/bin/FileCheck"
    echo "${EXPECTED_STAMP}" > "${INSTALL_STAMP_FILE}"
fi

restore_triton_sources() {
    if [ -f "${WS_TRITON_CLONE_PATH}/restore_sources.py" ]; then
        (cd "${WS_TRITON_CLONE_PATH}" && python3 restore_sources.py) || true
    fi
}
trap restore_triton_sources EXIT

echo "=== building triton-ascend ==="
cd "${WS_TRITON_CLONE_PATH}"

LLVM_SYSPATH="${WS_TRITON_LLVM_INSTALL_PATH}" \
TRITON_BUILD_WITH_CCACHE=true \
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_PROTON=OFF \
TRITON_WHEEL_NAME="triton-ascend" \
TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
python3 setup.py install
