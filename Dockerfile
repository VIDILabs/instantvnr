# Example commands:
#   docker build --build-arg CUDA_ARCH=86 -t instantvnr .
#   xhost +si:localuser:root
#   docker run --gpus all -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -w /instantvnr/build instantvnr

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Select a CUDA architecture to build. Currently we do not support multi-arch builds.
ARG CUDA_ARCH=90
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential mesa-utils pkg-config \
        libglx0 libglvnd0 libglvnd-dev \
        libgl1 libgl1-mesa-dev \
        libegl1 libegl1-mesa-dev \
        libgles2 libgles2-mesa-dev \
        libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libssl-dev \
        libaio-dev \
        wget git ninja-build imagemagick ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ADD https://raw.githubusercontent.com/NVlabs/nvdiffrec/main/docker/10_nvidia.json \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install CMake (3.24+ required by the standalone build).
RUN wget -qO- "https://cmake.org/files/v3.28/cmake-3.28.3-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install TBB
RUN wget -qO- "https://github.com/oneapi-src/oneTBB/releases/download/v2021.9.0/oneapi-tbb-2021.9.0-lin.tgz" | tar --strip-components=1 -xz -C /usr/local

WORKDIR /instantvnr
COPY . /instantvnr

# Configure and build the standalone project directly from this repository.
RUN SM=${CUDA_ARCH} BUILD_DIR=/instantvnr/build bash ./setup_cmake.sh

RUN BUILD_DIR=/instantvnr/build INSTALL_PREFIX=/instantvnr/install \
    bash ./setup_cmake.sh --install

RUN ln -s /instantvnr/data /instantvnr/build/data \
    && cp /instantvnr/example-model.json /instantvnr/build/example-model.json

ENV CMAKE_PREFIX_PATH=/instantvnr/install

WORKDIR /instantvnr/build
