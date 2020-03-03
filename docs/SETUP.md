## Setting up Environment

Please follow these instructions to read about the preferred compute environment and to set up the environment.

### Compute environment

The code in this repo has been developed and tested on an Azure Linux VM. Therefore, we recommend using an [Azure Data Science Virtual Machine (DSVM) for Linux (Ubuntu)](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to run the example notebooks and scripts. This VM will come installed with all the system requirements that are needed to create the conda environment described below and then run the notebooks in this repository. 

### Clone the repository
To clone the Forecasting repository to your local machine, please run:

```
git clone https://github.com/microsoft/forecasting.git
cd forecasting/
```

Next, follow the instruction below to install all dependencies required to run the examples provided in the repository. Follow [Automated environment setup](#automated-environment-setup) section to setup the environment automatically using a script. Alternatively, follow the [Manual environment setup](#manual-environment-setup) section for a step-by-step guide to setting up the environment.

### Automated environment setup

We provide a script to install all dependencies automatically on a Linux machine. To execute the script, please run: 

```
./tools/environment_setup.sh
```
from the root of Forecasting repo. If you have issues with running the setup script, please follow the [Manual environment setup](#manual-environment-setup) instructions below. 

Once you've executed the setup script, you can run example notebooks under [examples/](./examples) directory.


### Manual environment setup
#### Conda environment

To install the package contained in this repository, navigate to the directory where you pulled the Forecasting repo to run:
```bash
conda update conda
conda env create -f tools/environment.yaml
```
This will create the appropriate conda environment to run experiments. Next activate the installed environment:
```bash
conda activate forecasting_env
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

#### Jupyter kernel
In order to run the example notebooks, make sure to run the notebooks in the conda environment we previously set up, `forecasting_env`. To register the conda environment in Jupyter, please run:

```
python -m ipykernel install --user --name forecasting_env
```

Once you've set up the environment, you can run example notebooks under [examples/](./examples) directory.


