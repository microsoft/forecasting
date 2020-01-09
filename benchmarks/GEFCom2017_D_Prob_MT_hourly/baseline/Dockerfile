## Download base image
FROM rocker/r-base
ADD ./conda_dependencies.yml /tmp
ADD ./install_R_dependencies.R /tmp
WORKDIR /tmp

## Install basic packages
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    wget \
    zlib1g-dev \
    libssl-dev \
    libssh2-1-dev \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libsqlite3-dev \
    libc6-dev \
    libbz2-dev \
    libffi-dev \
    bzip2 \
    build-essential \
    checkinstall \
    ca-certificates \
    lsb-release \
    apt-utils \
    python3-pip \
    vim

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p $HOME/miniconda
ENV PATH="/root/miniconda/bin:${PATH}"

## Create conda environment
RUN conda update -y conda
RUN conda env create --file conda_dependencies.yml

# Install prerequisites of R packages
RUN apt-get install -y \
    gfortran \
    liblapack-dev \
    liblapack3 \
    libopenblas-base \
    libopenblas-dev \
    g++
## Mount R dependency file into the docker container and install dependencies
# Use a MRAN snapshot URL to download packages archived on a specific date
RUN echo 'options(repos = list(CRAN = "http://mran.revolutionanalytics.com/snapshot/2018-09-01/"))' >> /etc/R/Rprofile.site
RUN Rscript install_R_dependencies.R

RUN rm install_R_dependencies.R
RUN rm conda_dependencies.yml

RUN mkdir /Forecasting
WORKDIR /Forecasting

ENTRYPOINT ["/bin/bash"]
