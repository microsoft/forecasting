# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import pandas as pd
from git import Repo
from pkgutil import iter_modules


def git_repo_path():
    """Return the path of the forecasting repo"""

    repo = Repo(search_parent_directories=True)
    return repo.working_dir


def module_exists(module_name):
    """Check if a package is installed"""
    return module_name in (name for loader, name, ispkg in iter_modules())


# Source repo:
# https://github.com/Azure/MachineLearningNotebooks/blob/master/\
# how-to-use-azureml/automated-machine-learning/forecasting-orange-\
# juice-sales/forecasting_helper.py
def align_outputs(
    y_predicted,
    X_trans,
    X_test,
    y_test,
    target_column_name,
    predicted_column_name="predicted",
    horizon_colname="horizon_origin",
):
    """
    Demonstrates how to get the output aligned to the inputs
    using pandas indexes. Helps understand what happened if
    the output's shape differs from the input shape, or if
    the data got re-sorted by time and grain during forecasting.


    Typical causes of misalignment are:
    * we predicted some periods that were missing in actuals -> drop from eval
    * model was asked to predict past max_horizon -> increase max horizon
    * data at start of X_test was needed for lags -> provide previous periods
    """

    if horizon_colname in X_trans:
        df_fcst = pd.DataFrame({predicted_column_name: y_predicted, horizon_colname: X_trans[horizon_colname]})
    else:
        df_fcst = pd.DataFrame({predicted_column_name: y_predicted})

    # y and X outputs are aligned by forecast() function contract
    df_fcst.index = X_trans.index

    # align original X_test to y_test
    X_test_full = X_test.copy()
    X_test_full[target_column_name] = y_test

    # X_test_full's index does not include origin, so reset for merge
    df_fcst.reset_index(inplace=True)
    X_test_full = X_test_full.reset_index().drop(columns="index")
    together = df_fcst.merge(X_test_full, how="right")

    # drop rows where prediction or actuals are nan
    # happens because of missing actuals
    # or at edges of time due to lags/rolling windows
    clean = together[together[[target_column_name, predicted_column_name]].notnull().all(axis=1)]
    return clean
