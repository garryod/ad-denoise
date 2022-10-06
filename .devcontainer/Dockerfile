FROM registry.hub.docker.com/nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG PYTHON_VERSION=3.10
ARG PYTHON_PATH=/usr/local/python
ARG INSTALL_ZSH=true
ENV PIPX_HOME=/usr/local/py-utils \
    PIPX_BIN_DIR=${PIPX_HOME}/bin \
    PATH=${PYTHON_PATH}/bin:${PATH}:${PIPX_BIN_DIR}

COPY .devcontainer/library-scripts/* /tmp/library-scripts/

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y wget \
    && bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" automatic automatic automatic false "${INSTALL_ZSH}" false \
    && bash /tmp/library-scripts/python-debian.sh "${PYTHON_VERSION}" "${PYTHON_PATH}" "${PIPX_HOME}" automatic true true true true \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts/
