WS_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_WORKSPACE_PATH="$(cd "${WS_SCRIPTS_DIR}/.." && pwd)"
WS_REPOS_PATH="${WS_WORKSPACE_PATH}/repos"
WS_EXAMPLES_PATH="${WS_WORKSPACE_PATH}/examples"
WS_VENV_PATH="${WS_WORKSPACE_PATH}/venv"
WS_DUMMY_BIN_PATH="${WS_SCRIPTS_DIR}/bin"

WS_BUILD_DIR_NAME="build"
WS_INSTALL_DIR_NAME="install"
WS_OUTPUT_DIR_NAME="output"

WS_TRITON_LLVM_CLONE_PATH="${WS_REPOS_PATH}/llvm"
WS_TRITON_CLONE_PATH="${WS_REPOS_PATH}/triton"
WS_NPUIR_CLONE_PATH="${WS_REPOS_PATH}/npuir"

WS_TRITON_LLVM_INSTALL_PATH="${WS_TRITON_LLVM_CLONE_PATH}/${WS_BUILD_DIR_NAME}/${WS_INSTALL_DIR_NAME}"
WS_NPUIR_INSTALL_PATH="${WS_NPUIR_CLONE_PATH}/bin"

WS_CANN_PATH="${WS_CANN_PATH:-/usr/local/Ascend/cann-9.2.0}"
WS_ASCEND_SET_ENV="${WS_ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
WS_BISHENG_INSTALL_PATH="${WS_CANN_PATH}/tools/ccec_compiler/bin"
WS_BISHENG_CC1_PATH="${WS_CANN_PATH}/tools/bisheng_compiler/bin"

WS_CC=/usr/bin/clang-15
WS_CXX=/usr/bin/clang++-15
WS_LINKER=/usr/bin/lld-15

if [ -z "${WS_ASCEND_ENV_LOADED:-}" ] && [ -f "${WS_ASCEND_SET_ENV}" ]; then
    export WS_ASCEND_ENV_LOADED=1
    . "${WS_ASCEND_SET_ENV}" >/dev/null 2>&1 || true
fi

export BISHENG_INSTALL_PATH="${WS_DUMMY_BIN_PATH}"

if [ -d "${WS_VENV_PATH}/bin" ]; then
    case ":${PATH}:" in
        *":${WS_VENV_PATH}/bin:"*) ;;
        *) export PATH="${WS_VENV_PATH}/bin:${PATH}" ;;
    esac
fi

case ":${PATH}:" in
    *":${WS_NPUIR_INSTALL_PATH}:"*) ;;
    *) export PATH="${WS_NPUIR_INSTALL_PATH}:${PATH}" ;;
esac

if [[ "${PATH}" != "${WS_DUMMY_BIN_PATH}:"* ]]; then
    export PATH="${WS_DUMMY_BIN_PATH}:${PATH//:${WS_DUMMY_BIN_PATH}/}"
fi

if [ -d "${WS_CANN_PATH}/x86_64-linux/lib64" ] && [ ! -f "${WS_CANN_PATH}/x86_64-linux/lib64/libhccl.so" ]; then
    if [ -f "${WS_CANN_PATH}/x86_64-linux/lib64/libhccl_v2.so" ]; then
        sudo ln -sf "${WS_CANN_PATH}/x86_64-linux/lib64/libhccl_v2.so" "${WS_CANN_PATH}/x86_64-linux/lib64/libhccl.so" 2>/dev/null || true
    fi
fi
