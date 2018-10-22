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
    python3-pip \
    vim

## Install R
ENV R_BASE_VERSION 3.5.1
RUN sh -c 'echo "deb http://cloud.r-project.org/bin/linux/ubuntu xenial-cran35/" >> /etc/apt/sources.list' \
    && gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9 \
    && gpg -a --export E084DAB9 | apt-key add -
RUN apt-get update && apt-get install -y --no-install-recommends r-base=${R_BASE_VERSION}-* \
    && echo 'options(repos = c(CRAN = "https://cloud.r-project.org"))' >> /etc/R/Rprofile.site

## Mount R dependency file into the docker container and install dependencies 
# Install prerequisites of 'forecast' package
RUN apt-get update && apt-get install -y \
    gfortran \
    libblas-dev \
    liblapack-dev 
# Use a MRAN snapshot URL to download packages archived on a specific date
RUN echo 'options(repos = list(CRAN = "http://mran.revolutionanalytics.com/snapshot/2018-08-27/"))' >> /etc/R/Rprofile.site
ADD ./install_R_dependencies.r /tmp
RUN Rscript install_R_dependencies.r
RUN rm ./install_R_dependencies.r

WORKDIR /
ENTRYPOINT ["/bin/bash"]