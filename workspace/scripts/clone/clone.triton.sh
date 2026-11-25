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

mkdir -p "${WS_REPOS_PATH}"

echo "=== cloning triton-ascend ==="
if [ ! -d "${WS_TRITON_CLONE_PATH}/.git" ]; then
    git clone -b afshin/fix-kirin9020 git@github.com:Huawei-CPLLab/triton-ascend.git "${WS_TRITON_CLONE_PATH}"
    git -C "${WS_TRITON_CLONE_PATH}" submodule update --init --depth 1 -- third_party/ascend/AscendNPU-IR
else
    echo "Triton-ascend already exists at ${WS_TRITON_CLONE_PATH}. Ensuring submodule is initialized."
    git -C "${WS_TRITON_CLONE_PATH}" submodule update --init --depth 1 -- third_party/ascend/AscendNPU-IR
fi

LLVM_HASH="$(tr -d '[:space:]' < "${WS_TRITON_CLONE_PATH}/cmake/llvm-hash.txt")"
EXPECTED_STAMP="$(compute_llvm_stamp "${WS_TRITON_CLONE_PATH}")"

echo "=== preparing llvm-project (${EXPECTED_STAMP}) ==="
if [ ! -d "${WS_TRITON_LLVM_CLONE_PATH}/.git" ]; then
    git init "${WS_TRITON_LLVM_CLONE_PATH}"
    git -C "${WS_TRITON_LLVM_CLONE_PATH}" remote add origin https://github.com/llvm/llvm-project.git
fi
SRC_STAMP_FILE="${WS_TRITON_LLVM_CLONE_PATH}/.git/triton_llvm_src_stamp"

if [ -f "${SRC_STAMP_FILE}" ] && [ "$(cat "${SRC_STAMP_FILE}")" = "${EXPECTED_STAMP}" ]; then
    echo "LLVM source already matches ${EXPECTED_STAMP}."
else
    if ! git -C "${WS_TRITON_LLVM_CLONE_PATH}" cat-file -e "${LLVM_HASH}^{commit}" 2>/dev/null; then
        git -C "${WS_TRITON_LLVM_CLONE_PATH}" fetch --depth 1 origin "${LLVM_HASH}"
    fi
    git -C "${WS_TRITON_LLVM_CLONE_PATH}" checkout -f "${LLVM_HASH}"
    git -C "${WS_TRITON_LLVM_CLONE_PATH}" reset --hard "${LLVM_HASH}"
    git -C "${WS_TRITON_LLVM_CLONE_PATH}" clean -fd

    PATCH_DIR="${WS_TRITON_CLONE_PATH}/third_party/ascend/patch"
    if [ -d "${PATCH_DIR}" ] && compgen -G "${PATCH_DIR}/llvm_patch_*.patch" > /dev/null; then
        while IFS= read -r patch_file; do
            echo "Applying $(basename "${patch_file}")..."
            git -C "${WS_TRITON_LLVM_CLONE_PATH}" apply "${patch_file}"
        done < <(LC_ALL=C printf '%s\n' "${PATCH_DIR}"/llvm_patch_*.patch | LC_ALL=C sort)
    fi
    echo "${EXPECTED_STAMP}" > "${SRC_STAMP_FILE}"
fi
