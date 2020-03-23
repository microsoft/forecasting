## Download base image
FROM ubuntu:16.04
WORKDIR /tmp

## Install basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    #python-pip \
    python3-pip \
    vim

# Update pip
#RUN pip install --upgrade pip
RUN pip3 install --upgrade pip

## Mount Python dependency file into the docker container and install dependencies 
WORKDIR /tmp
ADD ./python_dependencies.txt /tmp
#RUN pip install -r python_dependencies.txt
RUN pip3 install -r python_dependencies.txt

WORKDIR /
RUN rm -rf tmp
ENTRYPOINT ["/bin/bash"]