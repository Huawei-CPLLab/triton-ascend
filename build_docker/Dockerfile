FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -o Acquire::Retries=5 update && apt-get -o Acquire::Retries=5 install -y \
        git \ 
        wget \
        cmake \
        ninja-build \ 
        clang-15 \ 
        lld-15 \ 
        python3 \
        python3-pip \ 
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        ccache \
        build-essential \
        sudo \
        vim \
        fzf \
        tmux \
        tree \
        zlib1g-dev \
        libnuma-dev \
        pciutils \
        unzip \
        && rm -rf /var/lib/apt/lists/* && \
        update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
        update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=developer

RUN groupadd -g $GROUP_ID $USER_NAME && \
    useradd -l -m -u $USER_ID -g $GROUP_ID -s /bin/bash $USER_NAME && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY ./resources/cache/Ascend-cann-toolkit_9.2.0_linux-x86_64.run /tmp

RUN cd /tmp && \
    chmod +x Ascend-cann-toolkit_9.2.0_linux-x86_64.run && \
    ./Ascend-cann-toolkit_9.2.0_linux-x86_64.run --quiet --install --install-for-all && \
    rm ./Ascend-cann-toolkit_9.2.0_linux-x86_64.run && \
    if [ -f /usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64/libhccl_v2.so ] && [ ! -e /usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64/libhccl.so ]; then \
        ln -sf /usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64/libhccl_v2.so /usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64/libhccl.so; \
    fi && \
    if [ -f /usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64/libhccl_legacy.so ] && [ ! -e /usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64/libhccl_compat.so ]; then \
        ln -sf /usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64/libhccl_legacy.so /usr/local/Ascend/cann-9.2.0/x86_64-linux/lib64/libhccl_compat.so; \
    fi

COPY ./resources/cache/linux/native-linux-x64-6.0.0.47-Beta1.zip /opt

RUN cd /opt && \
    unzip native-linux-x64-6.0.0.47-Beta1.zip && \
    rm native-linux-x64-6.0.0.47-Beta1.zip
    
#COPY ./resources/knpu_runtime /opt/knpu_runtime

#COPY ./resources/libs/* /usr/local/Ascend/cann/lib64

RUN printf '%s\n' \
    'export PS1="\[\e[48;5;172m\][\D{%F}\[\e[m\e[48;5;39m\] \D{%T}]\[\e[m\e[48;5;235m\] \w \[\e[m\]\n\[\e[38;5;46m\]\u@\h \$\[\e[m\] "' \
    'export EDITOR=vim' \
    'shopt -s checkwinsize' \
    'shopt -s histappend' \
    'PROMPT_COMMAND='\''history -a'\''' \
    'alias l="ls -l"' \
    'alias la="ls -a"' \
    'alias ll="ls -la"' \
    'alias c="clear"' \
    'alias cdu="cd .."' \
    'alias cduu="cd ../.."' \
    'alias cduuu="cd ../../.."' \
    '' \
    'if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] ; then' \
    '    source /usr/local/Ascend/ascend-toolkit/set_env.sh' \
    'fi' \
    '' \
    'export VIRTUAL_ENV="/workspace/venv"' \
    'export PATH="${VIRTUAL_ENV}/bin:${PATH}"' \
    'if [ ! -d "$VIRTUAL_ENV/bin" ] ; then' \
    '    python3.11 -m venv $VIRTUAL_ENV' \
    '    pip install --no-cache-dir --upgrade pip setuptools wheel pybind11 ninja cmake PyYAML numpy requests' \
    '    pip install --no-cache-dir --upgrade torch==2.12.0+cpu --index-url https://download.pytorch.org/whl/cpu' \
    '    pip install --no-cache-dir --upgrade torch_npu==2.12.0rc1' \
    'fi' \
    >> /home/${USER_NAME}/.bashrc

RUN chown ${USER_ID}:${GROUP_ID} /home/${USER_NAME}/.bashrc
    
RUN printf '#!/bin/bash\n\
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] ; then\n\
    source /usr/local/Ascend/ascend-toolkit/set_env.sh\n\
fi\n\
if [ -f /workspace/venv/bin/activate ] ; then\n\
    source /workspace/venv/bin/activate\n\
fi\n\
exec "$@"\n' > /opt/entrypoint.sh && chmod +x /opt/entrypoint.sh

USER ${USER_NAME}

WORKDIR /workspace

ENTRYPOINT [ "/opt/entrypoint.sh" ]

CMD [ "/bin/bash" ]