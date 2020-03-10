# Forecasting Best Practices 

This repository contains examples and best practices for building Forecasting solutions and systems, provided as [Jupyter notebooks](examples) and [a library of utility functions](fclib). The focus of the repository is on state-of-the-art methods and common scenarios that are popular among researchers and practitioners working on forecasting problems.

## Getting Started

To quickly get started with the repository on your local machine, use the following commands.

1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.

2. Clone the repository
    ```
    git clone https://github.com/microsoft/forecasting
    ```
3. Create and activate a conda environment
    ```
    cd forecasting
    conda env create -f ./tools/environment.yml
    conda activate forecasting_env
    ```
4. Install forecasting utilities
    ```
    pip install -e fclib
    ```
4. Register conda environment with Jupyter:
    ```
    python -m ipykernel install --user --name forecasting_env
    ```
5. Start the Jupyter notebook server
    ```
    jupyter notebook
    ```
6. Run the [LightGBM single-round](examples/00_quick_start/lightgbm_single_round.ipynb) notebook under the `00_quick_start` folder. Make sure that the selected Jupyter kernel is `forecasting_env`.

For detailed instructions on how to set up your environment and run examples provided in the repository, on local or a remote machine, please navigate to the [Setup Guide](./SETUP.md).

## Contributing
We hope that the open source community would contribute to the content and bring in the latest SOTA algorithm. This project welcomes contributions and suggestions. Before contributing, please see our [Contributing Guide](./docs/CONTRIBUTING.md).

## Build Status
| Build | Branch | Status |
| --- | --- | --- |
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/cpu_unit_tests_linux?branchName=master)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=128&branchName=master) |
| **Linux CPU** | staging | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/cpu_unit_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=128&branchName=staging) |