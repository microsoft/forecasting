# Forecasting Best Practices 

Time series forecasting is one of the most important topics in data science. Almost every business needs to predict the future in order to make better decisions and allocate resources more effectively.

This repository provides examples and best practice guidelines for building forecasting solutions. The goal of this repository is to build a comprehensive set of tools and examples that leverage recent advances in forecasting algorithms to build solutions and operationalize them. Rather than creating implementions from scratch, we draw from existing state-of-the-art libraries and build additional utility around processing and featurizing the data, optimizing and evaluating models, and scaling up to the cloud. 

This repository contains examples and best practices for building forecasting solutions and systems, provided as [Jupyter notebooks and R markdown files](examples) and [a library of utility functions](fclib). We hope that these examples and utilities can significantly reduce the “time to market” by simplifying the experience from defining the business problem to development of solution by orders of magnitude. In addition, the example notebooks would serve as guidelines and showcase best practices and usage of the tools in a wide variety of languages.

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
6. Run the [LightGBM single-round](examples/oj_retail/python/00_quick_start/lightgbm_single_round.ipynb) notebook under the `00_quick_start` folder. Make sure that the selected Jupyter kernel is `forecasting_env`.

For detailed instructions on how to set up your environment and run examples provided in the repository, on local or a remote machine, please navigate to the [Setup Guide](./SETUP.md).

## Target Audience
Our target audience for this repository includes data scientists and machine learning engineers with varying levels of knowledge in forecasting as our content is source-only and targets custom machine learning modelling. The utilities and examples provided are intended to be solution accelerators for real-world forecasting problems.

## Contributing
We hope that the open source community would contribute to the content and bring in the latest SOTA algorithm. This project welcomes contributions and suggestions. Before contributing, please see our [Contributing Guide](./docs/CONTRIBUTING.md).

## Build Status
| Build | Branch | Status |
| --- | --- | --- |
| **Linux CPU** | master | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/cpu_unit_tests_linux?branchName=master)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=128&branchName=master) |
| **Linux CPU** | staging | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/cpu_unit_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=128&branchName=staging) |