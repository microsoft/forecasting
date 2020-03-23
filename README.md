# Forecasting Best Practices 

Time series forecasting is one of the most important topics in data science. Almost every business needs to predict the future in order to make better decisions and allocate resources more effectively.

This repository provides examples and best practice guidelines for building forecasting solutions. The goal of this repository is to build a comprehensive set of tools and examples that leverage recent advances in forecasting algorithms to build solutions and operationalize them. Rather than creating implementations from scratch, we draw from existing state-of-the-art libraries and build additional utilities around processing and featurizing the data, optimizing and evaluating models, and scaling up to the cloud. 

The examples and best practices are provided as [Python Jupyter notebooks and R markdown files](examples) and [a library of utility functions](fclib). We hope that these examples and utilities can significantly reduce the “time to market” by simplifying the experience from defining the business problem to the development of solutions by orders of magnitude. In addition, the example notebooks would serve as guidelines and showcase best practices and usage of the tools in a wide variety of languages.


## Content

The following is a summary of the examples related to the process of building forecasting solutions covered in this repository. The [examples](examples) are organized according to use cases. Currently, we focus on a retail sales forecasting use case as it is widely used in [assortment planning](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1569&context=edissertations), [inventory optimization](https://en.wikipedia.org/wiki/Inventory_optimization), and [price optimization](https://en.wikipedia.org/wiki/Price_optimization).

| Example                          | Models/Methods                                        | Description                                                                                                                  | Language  |
|----------------------------------|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|-----------|
| Quick Start                      | Auto ARIMA, Azure AutoML, Linear Regression, LightGBM | Quick start notebooks that demonstrate workflow of developing a forecast model using one-round training and testing data     | Python    |
| Data Exploration and Preparation | Statistical Analysis and Data Transformation          | Data exploration and preparation examples                                                                                    | Python, R |
| Model Training and Evaluation    | Auto ARIMA, LightGBM, Dilated CNN                     | Deep dive notebooks that perform multi-round training and testing of various classical and deep learning forecast algorithms | Python    |
| Model Tuning and Deployment      | HyperDrive, LightGBM                                  | Example notebook for model tuning using Azure Machine Learning Service and deploying the best model on Azure                 | Python    |
| R Models                         | Mean Forecast, ARIMA, ETS, Prophet                    | Popular statistical forecast models and Prophet model implmented in R                                                        | R         |


## Getting Started in Python

To quickly get started with the repository on your local machine, use the following commands.

1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.

2. Clone the repository
    ```
    git clone https://github.com/microsoft/forecasting
    cd forecasting/
    ```

3. Run setup scripts to create conda environment. Please execute one of the following commands from the root of Forecasting repo based on your operating system.

    - Linux
    ```
    ./tools/environment_setup.sh
    ```

    - Windows
    ```
    tools\environment_setup.bat
    ```

    Note that for Windows you need to run the batch script from Anaconda Prompt. The script creates a conda environment `forecasting_env` and installs the forecasting utility library `fclib`.

4. Start the Jupyter notebook server
    ```
    jupyter notebook
    ```
    
5. Run the [LightGBM single-round](examples/oj_retail/python/00_quick_start/lightgbm_single_round.ipynb) notebook under the `00_quick_start` folder. Make sure that the selected Jupyter kernel is `forecasting_env`.

If you have any issues with the above setup, or want to find more detailed instructions on how to set up your environment and run examples provided in the repository, on local or a remote machine, please navigate to the [Setup Guide](./docs/SETUP.md).

## Getting Started in R

We assume you already have R installed on your machine. If not, simply follow the [instructions on CRAN](https://cloud.r-project.org/) to download and install R.

The recommended editor is [RStudio](https://rstudio.com), which supports interactive editing and previewing of R notebooks. However, you can use any editor or IDE that supports RMarkdown. In particular, [Visual Studio Code](https://code.visualstudio.com) with the [R extension](https://marketplace.visualstudio.com/items?itemName=Ikuyadeu.r) can be used to edit and render the notebook files. The rendered `.nb.html` files can be viewed in any modern web browser.

The examples use the [Tidyverts](https://tidyverts.org) family of packages, which is a modern framework for time series analysis that builds on the widely-used [Tidyverse](https://tidyverse.org) family. The Tidyverts framework is still under active development, so it's recommended that you update your packages regularly to get the latest bug fixes and features.

## Target Audience
Our target audience for this repository includes data scientists and machine learning engineers with varying levels of knowledge in forecasting as our content is source-only and targets custom machine learning modelling. The utilities and examples provided are intended to be solution accelerators for real-world forecasting problems.

## Contributing
We hope that the open source community would contribute to the content and bring in the latest SOTA algorithm. This project welcomes contributions and suggestions. Before contributing, please see our [Contributing Guide](./docs/CONTRIBUTING.md).

## Reference

The following is a list of related repositories that you may find helpful.

|                                                                                                                        |                                                                                                 |
|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| [Deep Learning for Time Series Forecasting](https://github.com/Azure/DeepLearningForTimeSeriesForecasting)             | A collection of examples for using deep neural networks for time series forecasting with Keras. |
| [Demand Forecasting and Price Optimization Solution](https://github.com/Azure/cortana-intelligence-price-optimization) | A Cortana Intelligence solution how-to guide for demand forecasting and price optimization.     |



## Build Status
| Build         | Branch  | Status                                                                                                                                                                                                                             |
|---------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Linux CPU** | master  | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/cpu_unit_tests_linux?branchName=master)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=128&branchName=master)   |
| **Linux CPU** | staging | [![Build Status](https://dev.azure.com/best-practices/forecasting/_apis/build/status/cpu_unit_tests_linux?branchName=staging)](https://dev.azure.com/best-practices/forecasting/_build/latest?definitionId=128&branchName=staging) |
