## Setting up Environment

Please follow these instructions to read about the preferred compute environment and to set up the environment. 

### Compute environment

The code in this repo has been developed and tested on an Azure Linux VM. Therefore, we recommend using an [Azure Data Science Virtual Machine (DSVM) for Linux (Ubuntu)](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro) to run the example notebooks and scripts. This VM will come installed with all the system requirements that are needed to create the conda environment described below and then run the notebooks in this repository. If you are using a Linux machine without conda installed, please install Miniconda by following the instructions in this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

You can also use a Windows machine to run the example notebooks and scripts. In this case, you may either work with a [Windows Server 2019 Data Science Virtual Machine on Azure](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/provision-vm) or a local Windows machine. Azure Windows VW comes with conda pre-installed. If conda is not installed on your machine, please follow the instructions in this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) to install Miniconda.

### Clone the repository

To clone the Forecasting repository to your local machine, please run:

```
git clone https://github.com/microsoft/forecasting.git
cd forecasting/
```

Next, follow the instruction below to install all dependencies required to run the examples provided in the repository. Follow [Automated environment setup](#automated-environment-setup) section to set up the environment automatically using a script. Alternatively, follow the [Manual environment setup](#manual-environment-setup) section for a step-by-step guide to setting up the environment.

### Automated environment setup

We provide scripts to install all dependencies automatically on a Linux machine as well as on a Windows machine. 

#### Linux

If you are using a Linux machine, please run the following command to execute the shell script for Linux
```
./tools/environment_setup.sh
```
from the root of Forecasting repo. 

#### Windows

Similarly, if you are using a Windows machine, please run the batch script for Windows via
```
tools\environment_setup.bat
```
from the root of Forecasting repo. Note that you need to run the above command from Anaconda Prompt (a terminal with conda available), which can be started by opening the Windows Start menu and clicking `Anaconda Prompt (Miniconda3)` as follows

<p align="center">
<img src="https://user-images.githubusercontent.com/20047467/76897869-f2f22900-686a-11ea-9f67-b189c15df27a.png" width="210" height="395">
</p>

Once you've executed the setup script, please activate the newly created conda environment:

```
conda activate forecasting_env
```

>!NOTE: If you have issues with running the setup script, please follow the [Manual environment setup](#manual-environment-setup) instructions below. 

Next, navigate to [Starting the Jupyter Notebook Server](#starting-the-jupyter-notebook-server) section below to start the Jupyter server necessary for running the examples.


### Manual environment setup

#### Conda environment

To install the package contained in this repository, navigate to the directory where you pulled the Forecasting repo to run:
```bash
conda update conda
conda env create -f tools/environment.yml
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

If you work with Linux or MacOS, you could also install Ray for parallel model training:
```
pip install ray>=0.8.2 
``` 

#### Jupyter kernel
In order to run the example notebooks, make sure to run the notebooks in the conda environment we previously set up, `forecasting_env`. To register the conda environment in Jupyter, please run:

```
python -m ipykernel install --user --name forecasting_env
```

### Starting the Jupyter Notebook Server
In order to run the example notebooks provided in this repository, you will have to start a Jupyter notebook server. 

For running examples on your **local machine**, please open your terminal application and run the following command:

```
jupyter notebook
```

If you are working on a remote VM, you can start the notebook server with the following command:
```
jupyter notebook --no-browser --port=8889
``` 
and forward the port where the notebooks are running (e.g., 8889) to the local machine via running the following command from the local machine:
```
ssh -L localhost:8889:localhost:8889 <user-name>@<ip-address-of-the-vm>
```

To access the notebooks, type `localhost:8889/` in the browser on your local machine.

Now you're ready to run the examples provided in the `examples/`, by simply opening and executing the notebooks in the Jupyter server. Please also navigate to the [examples README file](../examples/README.md) to read about the available notebooks. 
