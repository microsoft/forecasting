## Download base image
FROM continuumio/anaconda3:4.4.0
ADD TSPerf/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/conda_dependencies.yml /tmp/conda_dependencies.yml
ADD TSPerf/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/install_R_dependencies.R /tmp/install_R_dependencies.R
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

## Create and activate conda environment
RUN conda update -y conda
RUN conda update -y conda
RUN conda env create --file conda_dependencies.yml

## Install R
ENV R_BASE_VERSION 3.5.1
RUN apt-get install -y aptitude
RUN echo "deb http://http.debian.net/debian sid main" > /etc/apt/sources.list.d/debian-unstable.list \
        && aptitude install -y debian-keyring debian-archive-keyring
RUN apt-get remove -y binutils
RUN apt-get update \
        && apt-get install -t unstable -y --no-install-recommends \
                r-base=${R_BASE_VERSION}-*

# Install prerequisites of R packages
RUN apt-get install -y \
    gfortran \
    liblapack-dev \
    liblapack3 \
    libopenblas-base \
    libopenblas-dev
## Mount R dependency file into the docker container and install dependencies
# Use a MRAN snapshot URL to download packages archived on a specific date
RUN echo 'options(repos = list(CRAN = "http://mran.revolutionanalytics.com/snapshot/2018-09-01/"))' >> /etc/R/Rprofile.site
RUN Rscript install_R_dependencies.R

RUN rm install_R_dependencies.R
RUN rm conda_dependencies.yml

RUN mkdir /TSPerf
WORKDIR /TSPerf

ENTRYPOINT ["/bin/bash"]