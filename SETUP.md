## Setting up Environment

Follow the instruction bellow to read about compute requirements and install required libraries.


#### Compute environment

We recommend using a virtual machine to run the example notebooks and scripts. The easiest way to get started is to use the [Azure Data Science Virtual Machine (DSVM) for Linux (Ubuntu)](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro). This VM will come installed with all the system requirements that are needed to create the conda environment described below and then run the notebooks in this repository. 

#### Conda environment

To install the package contained in this repository, navigate to the directory where you pulled the Forecasting repo to run:
```bash
conda update conda
conda env create -f tools/environment.yaml
```
This will create the appropriate conda environment to run experiments. Next activate the installed environment:
```bash
conda activate forecast
```

During development, in case you need to update the environment due to a conda env file change, you can run
```
conda env update --file tools/environment.yaml
```
from the root of Forecasting repo.

#### Package installation

Next you will need to install the common package for forecasting:
```bash
pip install -e fclib
```

The library is installed in developer mode with the `-e` flag. This means that all changes made to the library locally, are immediately available.

## Getting Started with Examples

### Dataset

This repository provides examples on how to run various forecasting algorithms on a publicly available `orangeJuice` dataset. This dataset containes weekly sales of refrigerated orange juice at 83 stores, as well as, demographic information on those stores. The data is provided via [`bayesm`](https://CRAN.R-project.org/package=bayesm) package in `R`, and further described in [Chapter 5, Bayesian Statistics and Marketing by Rossi, Allenby, and McCulloch](http://www.perossi.org/home/bsm-1). `fclib` package provides utilies for downloading and processing this data. 

### Run Examples

#### Notebooks
We provide example notebooks under `examples/` to demonstrate how to train and evaluate forecasting algorithms.

Make sure to run the notebooks in the conda environment we previously set up (`forecast`). To register the conda environment in Jupyter, please run:

```
python -m ipykernel install --user --name forecast
```