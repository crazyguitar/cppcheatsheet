# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Modifications copyright (c) 2025 Chang-Ning Tsai

# ref: https://github.com/aws-samples/awsome-distributed-training/blob/main/micro-benchmarks/nccl-tests
ARG CUDA_VERSION=12.8.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ARG GDRCOPY_VERSION=v2.5.1
ARG EFA_INSTALLER_VERSION=1.46.0
ARG AWS_OFI_NCCL_VERSION=v1.16.3
ARG NCCL_VERSION=v2.29.2-1
ARG NCCL_TESTS_VERSION=v2.17.7
ARG CUBLASDX_VERSION=25.06.0

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get remove -y --allow-change-held-packages \
    ibverbs-utils \
    libibverbs-dev \
    libibverbs1 \
    libmlx5-1 \
    libnccl2 \
    libnccl-dev

RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

ENV OPAL_PREFIX=

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    apt-utils \
    autoconf \
    automake \
    build-essential \
    check \
    cmake \
    ninja-build \
    curl \
    debhelper \
    devscripts \
    git \
    gcc \
    gdb \
    kmod \
    libsubunit-dev \
    libtool \
    openssh-client \
    openssh-server \
    pkg-config \
    python3-distutils \
    vim \
    hwloc \
    libhwloc-dev \
    python3.10-dev \
    python3.10-venv \
    libboost-all-dev \
    libomp-dev

#################################################
## Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/kitware.list && \
    apt-get update && \
    apt-get install -y cmake

RUN apt-get purge -y cuda-compat-*

RUN mkdir -p /var/run/sshd
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# Set paths for both aarch64 and x86_64
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib/aarch64-linux-gnu:/opt/amazon/ofi-nccl/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:/usr/bin:/usr/local/bin:$PATH

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py \
    && pip3 install awscli pynvml Cython

#################################################
## Install NVIDIA GDRCopy
##
## NOTE: if `nccl-tests` or `/opt/gdrcopy/bin/sanity -v` crashes with incompatible version, ensure
## that the cuda-compat-xx-x package is the latest.
RUN git clone -b ${GDRCOPY_VERSION} https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy \
    && make prefix=/opt/gdrcopy install

ENV LD_LIBRARY_PATH=/opt/gdrcopy/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/gdrcopy/lib:$LIBRARY_PATH
ENV CPATH=/opt/gdrcopy/include:$CPATH
ENV PATH=/opt/gdrcopy/bin:$PATH

#################################################
## Install EFA installer
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf $HOME/aws-efa-installer

###################################################
## Install NCCL
RUN git clone -b ${NCCL_VERSION} https://github.com/NVIDIA/nccl.git  /opt/nccl \
    && cd /opt/nccl \
    && make -j $(nproc) src.build CUDA_HOME=/usr/local/cuda \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

###################################################
## Install NCCL-tests
RUN git clone -b ${NCCL_TESTS_VERSION} https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j $(nproc) \
    MPI=1 \
    MPI_HOME=/opt/amazon/openmpi/ \
    CUDA_HOME=/usr/local/cuda \
    NCCL_HOME=/opt/nccl/build \
    NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100"

###################################################
## Install NVSHMEM
ENV NVSHMEM_DIR=/opt/nvshmem
ENV NVSHMEM_HOME=/opt/nvshmem
RUN git clone https://github.com/NVIDIA/nvshmem.git /nvshmem \
  && cd /nvshmem \
  && git checkout 131da55f643ac87c810ba0bc51d359258bf433a1 \
  && mkdir -p build \
  && cd build \
  && cmake -DNVSHMEM_PREFIX=/opt/nvshmem \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DNVSHMEM_MPI_SUPPORT=1 \
    -DNVSHMEM_PMIX_SUPPORT=1 \
    -DNVSHMEM_LIBFABRIC_SUPPORT=1 \
    -DNVSHMEM_IBRC_SUPPORT=1 \
    -DNVSHMEM_IBGDA_SUPPORT=1 \
    -DNVSHMEM_USE_GDRCOPY=1 \
    -DNVSHMEM_BUILD_TESTS=1 \
    -DNVSHMEM_BUILD_EXAMPLES=1 \
    -DNVSHMEM_BUILD_HYDRA_LAUNCHER=1 \
    -DNVSHMEM_BUILD_TXZ_PACKAGE=0 \
    -DNVSHMEM_BUILD_PYTHON_LIB=0 \
    -DMPI_HOME=/opt/amazon/openmpi \
    -DPMIX_HOME=/opt/amazon/pmix \
    -DGDRCOPY_HOME=/opt/gdrcopy \
    -DLIBFABRIC_HOME=/opt/amazon/efa \
    -G Ninja .. \
  && ninja -j $(nproc) \
  && ninja install

# install nvshmemrun
RUN cd /nvshmem/scripts && bash install_hydra.sh /tmp /opt/hydra && rm -rf /tmp/hydra-*
RUN pip3 install nvshmem4py-cu12

ENV LD_LIBRARY_PATH=/opt/amazon/pmix/lib:/opt/nvshmem/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/nvshmem/bin:$PATH
ENV NVSHMEM_REMOTE_TRANSPORT=libfabric
ENV NVSHMEM_LIBFABRIC_PROVIDER=efa

###################################################
## Install CUBLASDx
RUN cd ${HOME} \
  && wget https://developer.download.nvidia.com/compute/cublasdx/redist/cublasdx/nvidia-mathdx-${CUBLASDX_VERSION}.tar.gz \
  && tar -zvxf nvidia-mathdx-${CUBLASDX_VERSION}.tar.gz \
  && mv nvidia-mathdx-${CUBLASDX_VERSION}/nvidia/mathdx/25.06 /opt/nvidia/mathdx \
  && rm -rf nvidia-mathdx-${CUBLASDX_VERSION}.tar.gz  nvidia-mathdx-${CUBLASDX_VERSION}

ENV MATHDX_ROOT=/opt/nvidia/mathdx
ENV CMAKE_PREFIX_PATH=$MATHDX_ROOT:$CMAKE_PREFIX_PATH

RUN rm -rf /var/lib/apt/lists/*

## Set Open MPI variables to exclude network interface and conduit.
ENV OMPI_MCA_pml=^ucx            \
    OMPI_MCA_btl=tcp,self           \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent\
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth

## Turn off PMIx Error https://github.com/open-mpi/ompi/issues/7516
ENV PMIX_MCA_gds=hash

## Set LD_PRELOAD for NCCL library
ENV LD_PRELOAD=/opt/nccl/build/lib/libnccl.so
