#!/bin/bash

set -e

USER_NAME=$(whoami)
IMAGE_NAME="triton-ascend-${USER_NAME}"

mkdir -p workspace

docker run -it --rm --hostname KIRIN_DOCKER \
    -v ~/.gitconfig:/home/${USER_NAME}/.gitconfig \
    -v ~/.ssh:/home/${USER_NAME}/.ssh \
    -v $(pwd)/workspace:/workspace ${IMAGE_NAME}
    