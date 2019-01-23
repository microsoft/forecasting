## Download base image
FROM continuumio/anaconda3:4.4.0
ADD ./conda_dependencies.yml /tmp
WORKDIR /tmp

## Install basic packages
RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
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
    vim 

## Create and activate conda environment
RUN conda update -y conda
RUN conda env create --file conda_dependencies.yml

RUN rm conda_dependencies.yml

RUN mkdir /TSPerf
WORKDIR /TSPerf

ENTRYPOINT ["/bin/bash"]