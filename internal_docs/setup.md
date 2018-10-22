### Guide to setup TSPerf environment

## Prerequisites

To run scripts provided by the TSPerf framework, you must install:

- [Anaconda Python 3.6 version](https://www.anaconda.com/download/)
or
- [Miniconda Python 3.6](https://conda.io/miniconda.html)

## Environment setup

Run the following commands to set up the tsperf conda environment:
```
cd TSPerf
conda env create -f common/conda_dependencies.yml
source activate tsperf
```
If you would like to use this environment in a jupyter notebook, run the following:
```
python -m ipykernel install --user --name tsperf --display-name "tsperf"
```

