#!/bin/bash

set -e

USER_NAME=$(whoami)
IMAGE_NAME="triton-ascend-${USER_NAME}"

RESOURCE_DIR="./resources/cache"
CANN_BIN="Ascend-cann-toolkit_9.2.0_linux-x86_64.run"
CANN_LINK="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.2.0/Ascend-cann-toolkit_9.2.0_linux-x86_64.run"
OHOS_SDK_LINK="https://repo.huaweicloud.com/harmonyos/os/6.0-Release/ohos-sdk-windows_linux-public.tar.gz"
OHOS_SDK_TAR="ohos-sdk-windows_linux-public.tar.gz"
OHOS_SDK_LINUX="linux/native-linux-x64-6.0.0.47-Beta1.zip"

mkdir -p "${RESOURCE_DIR}"

if [ ! -f "${RESOURCE_DIR}/${CANN_BIN}" ]; then
    wget -O "${RESOURCE_DIR}/${CANN_BIN}" "${CANN_LINK}"
fi

if [ ! -f "${RESOURCE_DIR}/${OHOS_SDK_LINUX}" ]; then
    pushd "${RESOURCE_DIR}"
    wget -O "${OHOS_SDK_TAR}" "${OHOS_SDK_LINK}"
    tar -xvzf "${OHOS_SDK_TAR}" ${OHOS_SDK_LINUX}
    rm -r ${OHOS_SDK_TAR}
    popd
fi

docker build \
    --build-arg USER_ID=$(id -u)\
    --build-arg GROUP_ID=$(id -g)\
    --build-arg USER_NAME=${USER_NAME}\
    -t "${IMAGE_NAME}" .    

