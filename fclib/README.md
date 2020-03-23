# Forecasting library

Building forecasting models can involve tedious tasks ranging from data loading, dataset understanding, model development, model evaluation to deployment of trained models. To assist with these tasks, we developed a forecasting library - **fclib**. You'll see this library used widely in sample notebooks in [examples](../examples). The following provides a short description of the sub-modules. For more details about what functions/classes/utitilies are available and how to use them, please review the doc-strings provided with the code and see the sample notebooks in [examples](../examples) directory.

## Submodules

### [AzureML](fclib/azureml)

The AzureML submodule contains utilities to connect to an Azure Machine Learning workspace, train, tune and operationalize forecasting models at scale using AzureML.


### [Common](fclib/common)

This submodule contains high-level utilities that are commonly used in multiple algorithms as well as helper functions for visualizing forecasting predictions.

### [Dataset](fclib/dataset)
This submodule includes helper functions for interacting with datasets used in the example notebooks, utility functions to process datasets for different models tasks, as well as utilities for splitting data for training/testing. For example, the [ojdata](fclib/dataset/ojdata.py) submodule will allow you to download and process Orange Juice data set, as well as split it into training and testing rounds. 

```python
from fclib.dataset.ojdata import download_ojdata, split_train_test

download_ojdata(DATA_DIR)
train_df_list, test_df_list, _ = split_train_test(
    DATA_DIR,
    n_splits=N_SPLITS,
    horizon=HORIZON,
    gap=GAP,
    first_week=FIRST_WEEK,
    last_week=LAST_WEEK
)
```

### [Evaluation](fclib/evaluation)
Evaluation module includes functionalities for computing common forecasting evaluation metrics, more specifically `MAPE`, `sMAPE`, and `pinball loss`.

### [Feature Engineering](fclib/feature_engineering)
Feature engineering module contains utilities to create various time series features, for example, week or day of month, lagged features, and moving average features. This module is used widely in machine-learning based approaches to forecasting, in which time series data is transformed into a tabular featurized dataset, that becomes input to a machine learning method.

### [Models](fclib/models)
The models module contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new forecasting solutions. Some submodules found here are: `lightgbm`, `dilated cnn`, etc. A more detailed description of which algorithms are used in our examples can be found in [this README](../examples/oj_retail/python/README.md).