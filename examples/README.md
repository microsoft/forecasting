# Forecasting examples

This folder contains Python examples for building forecasting solutions. To run the notebooks, please execute `jupyter notebook` and select the Jupyter kernel `forecasting_env` if you are using a local machine. Otherwise, if you use a remote VM, you can start the notebooks via `jupyter notebook --no-browser` and forward the port where the notebooks are running (e.g., 8888) to the local machine via `ssh <user-name>@<ip-address-of-the-vm> -L 8888:localhost:8888`.


## Summary

The following summarizes each directory of the best practice notebooks.

| Directory | Content | Description |
| --- | --- | --- |
| [00_quick_start](./00_quick_start)| [auto_arima_forecasting.ipynb](./00_quick_start/auto_arima_forecasting.ipynb) <br>[azure_automl_forecast.ipynb](./00_quick_start/azure_automl_forecast.ipynb) <br> [lightgbm_point_forecast.ipynb](./00_quick_start/lightgbm_point_forecast.ipynb) | Quick start notebooks that demonstrate workflow of developing a forecasting model using one-round training and testing data|
| [01_prepare_data](./01_prepare_data) | [ojdata_exploration_retail.ipynb](./01_prepare_data/ojdata_exploration_retail.ipynb) <br> [ojdata_preparation_retail.ipynb](./01_prepare_data/ojdata_preparation_retail.ipynb) | Data exploration and preparation notebooks|
| [02_model](./02_model) | [dilatedcnn_point_forecast_multiround.ipynb](./02_model/dilatedcnn_point_forecast_multiround.ipynb) <br> [lightgbm_point_forecast_multiround.ipynb](./02_model/lightgbm_point_forecast_multiround.ipynb) | Deep dive notebooks that perform multi-round training and testing of various classical and deep learning forecast algorithms|
| [03_model_select_deploy](03_model_select_deploy) | Example notebook to be added soon | Best practice notebook for model selecting by using Azure Machine Learning Service and deploying the best model on Azure|

