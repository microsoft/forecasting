# Forecasting examples in Python

This folder contains Jupyter notebooks with Python examples for building forecasting solutions. To run the notebooks, please ensure your environment is set up with required dependencies by following instructions in the [Setup guide](../../../docs/SETUP.md). 


## Summary

The following summarizes each directory of the Python best practice notebooks.

| Directory | Content | Description |
| --- | --- | --- |
| [00_quick_start](./00_quick_start)| [autoarima_single_round.ipynb](./00_quick_start/autoarima_single_round.ipynb) <br>[azure_automl_single_round.ipynb](./00_quick_start/azure_automl_single_round.ipynb) <br> [lightgbm_single_round.ipynb](./00_quick_start/lightgbm_single_round.ipynb) | Quick start notebooks that demonstrate workflow of developing a forecasting model using one-round training and testing data|
| [01_prepare_data](./01_prepare_data) | [ojdata_exploration.ipynb](./01_prepare_data/ojdata_exploration.ipynb) <br> [ojdata_preparation.ipynb](./01_prepare_data/ojdata_preparation.ipynb) | Data exploration and preparation notebooks|
| [02_model](./02_model) | [dilatedcnn_multi_round.ipynb](./02_model/dilatedcnn_multi_round.ipynb) <br> [lightgbm_multi_round.ipynb](./02_model/lightgbm_multi_round.ipynb) <br> [autoarima_multi_round.ipynb](./02_model/autoarima_multi_round.ipynb) | Deep dive notebooks that perform multi-round training and testing of various classical and deep learning forecast algorithms|
| [03_model_tune_deploy](./03_model_tune_deploy/) | [azure_hyperdrive_lightgbm.ipynb](./03_model_tune_deploy/azure_hyperdrive_lightgbm.ipynb) <br> [aml_scripts/](./03_model_tune_deploy/aml_scripts) | <ul><li> Example notebook for model tuning using Azure Machine Learning Service and deploying the best model on Azure </ul></li> <ul><li> Scripts for model training and validation </ul></li> |

