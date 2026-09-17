FROM ghcr.io/accel-sim/accel-sim-framework@sha256:c1f2ff7e773955f274dbeda259e69bf2f1722ef1bd39a541cabb9c6c63149dda

LABEL org.opencontainers.image.title="Accel-Sim 2.0 logging development"
LABEL org.opencontainers.image.description="Pinned Accel-Sim CUDA 12.9 environment with tracez build dependencies"
LABEL org.opencontainers.image.source="https://github.com/pcg108/accel-sim-framework"
LABEL org.opencontainers.image.version="2.0-logging"
LABEL accel-sim.framework.commit="64653015f85fb5664c84a10f48527e8897d289d0"
LABEL accel-sim.gpgpu-sim.commit="e10018b67a4b668e7b43f89280cf67624f1df4ff"
LABEL accel-sim.pybind11.commit="d4f9cfbc2866f2156e1b17cb478a67088c6063f6"

USER root

ARG DEBIAN_FRONTEND=noninteractive
# The pinned parent contains an expired/unsigned Nsight DevTools repository.
# None of the simulator dependencies come from it; retain the CUDA repository.
RUN rm -f /etc/apt/sources.list.d/nvidia.list \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libzstd-dev \
        ninja-build \
        pkg-config \
        xz-utils \
        zstd \
    && rm -rf /var/lib/apt/lists/*

ENV ACCELSIM_ROOT=/accel-sim/accel-sim-framework \
    GPGPUSIM_ROOT=/accel-sim/accel-sim-framework/gpu-simulator/gpgpu-sim \
    HF_HOME=/hf-cache \
    TRANSFORMERS_CACHE=/hf-cache/transformers \
    XDG_CACHE_HOME=/results/.cache

WORKDIR /accel-sim/accel-sim-framework

CMD ["/bin/bash"]
