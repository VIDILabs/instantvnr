# Example Command to Run
#   docker build -t instantvnr .
#   xhost +si:localuser:root
#   docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix instantvnr
#   docker run --runtime=nvidia -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ${PWD}:/instantvnr/source -w /instantvnr/build instantvnr

FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
        build-essential mesa-utils pkg-config \
        libglx0 libglvnd0 libglvnd-dev \
        libgl1 libgl1-mesa-dev \
        libegl1 libegl1-mesa-dev \
        libgles2 libgles2-mesa-dev \
        libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libssl-dev \
        libaio-dev \
        wget git ninja-build imagemagick
# RUN rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics
ADD https://raw.githubusercontent.com/NVlabs/nvdiffrec/main/docker/10_nvidia.json \
    /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install cmake
RUN wget -qO- "https://cmake.org/files/v3.23/cmake-3.23.2-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C /usr/local

# Install tbb
RUN wget -qO- "https://github.com/oneapi-src/oneTBB/releases/download/v2021.9.0/oneapi-tbb-2021.9.0-lin.tgz" | tar --strip-components=1 -xz -C /usr/local

# Create a superbuild
RUN git clone --recursive https://github.com/VIDILabs/open-volume-renderer.git /instantvnr/ovr
RUN git clone --recursive https://github.com/VIDILabs/instantvnr.git /instantvnr/source
RUN ln -s /instantvnr/source /instantvnr/ovr/projects/instantvnr

# Config and build
RUN mkdir -p /instantvnr/build
RUN cmake -S /instantvnr/ovr -B/instantvnr/build -GNinja \
    -DOptiX_INSTALL_DIR=/instantvnr/ovr/github-actions/optix-cmake-github-actions/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64 \
    -DGDT_CUDA_ARCHITECTURES=75 -DOVR_BUILD_MODULE_NNVOLUME=ON -DOVR_BUILD_DEVICE_OSPRAY=OFF -DOVR_BUILD_DEVICE_OPTIX7=ON
RUN cmake --build /instantvnr/build --config Release --parallel 16

RUN ln -s /instantvnr/ovr/data /instantvnr/build/data
RUN cp /instantvnr/source/example-model.json /instantvnr/build/example-model.json

WORKDIR [ '/instantvnr/build' ]
