## Download base image
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
WORKDIR /tmp

## Install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
    wget \
    zlib1g-dev \
    libssl-dev \
    libssh2-1-dev \
    libcurl4-openssl-dev \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    libffi-dev \
    bzip2 \
    build-essential \
    checkinstall \
    ca-certificates \
    curl \
    lsb-release \
    apt-utils \
    python3-pip \
    vim \
    cuda-command-line-tools-9-0 \
    cuda-cublas-9-0 \
    cuda-cufft-9-0 \
    cuda-curand-9-0 \
    cuda-cusolver-9-0 \
    cuda-cusparse-9-0 \
    libcudnn7=7.2.1.38-1+cuda9.0 \
    libnccl2=2.2.13-1+cuda9.0 \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng12-dev \
    libzmq3-dev 

RUN apt-get update && \
    apt-get install nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0 && \
    apt-get update && \
    apt-get install libnvinfer4=4.1.2-1+cuda9.0 

# Update pip and setuptools
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools

## Mount Python dependency file into the docker container and install dependencies 
WORKDIR /tmp
ADD ./python_dependencies.txt /tmp
RUN pip3 install -r python_dependencies.txt

# Fix the symlink issue of tensorflow (https://github.com/tensorflow/tensorflow/issues/10776)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH='/usr/local/cuda/lib64/stubs/:${LD_LIBRARY_PATH}'
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

WORKDIR /
RUN rm -rf tmp
ENTRYPOINT ["/bin/bash"]