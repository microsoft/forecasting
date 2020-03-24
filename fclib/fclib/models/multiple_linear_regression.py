# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. 

"""
This file contains utility functions for builing multiple linear regression 
models to forecast many time series individually.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression


def fit(train_df, grain_col_names, fea_col_names=[], target_col_name="target"):
    """Train multiple linear regression models with each being trained on an individual time 
    series specified by columns in grain_col_names.
    
    Args: 
        train_df (pandas.DataFrame): Training data frame including all the features
        grain_col_names (list[str]): List of the column names that specify each time series
        fea_col_names (list[str]): List of the names of columns that we want to use as input features
        target_col_name (str): Name of the target column
        
    Returns:
        dict: Dictionary including all the trained linear regression models
    """
    lr_models = {}
    if not fea_col_names:
        fea_col_names = list(train_df.columns)
        fea_col_names.remove(target_col_name)
    for name, group in train_df.groupby(grain_col_names):
        lr = LinearRegression()
        lr.fit(group[fea_col_names], group[target_col_name])
        lr_models[name] = lr
    return lr_models


def predict(
    test_df, lr_models, time_col_name, grain_col_names, fea_col_names=[], nonnegative_output=True, integer_output=True
):
    """Predict target variable with multiple linear regression models that have been trained.

    Args:
        test_df (pandas.DataFrame): Dataframe including all needed features
        lr_models (dict): A dictionary that includes all the trained linear regression models with format 
            {(grain1, grain2, ...): model1, (grain1, grain2, ...): model2, ...}
        time_col_name (str): Name of the time column
        grain_col_names (list[str]): List of the column names that specify each time series
        fea_col_names (list[str]): List of the names of the columns that are needed for generating predictions
        positive_output (bool): If it is True, negative forecasts will be replaced by 0
        integer_output (bool): If it is True, the forecast will be rounded to an integer

    Returns:
        pandas.DataFrame including the predictions of the target variable
    """
    pred_dfs = []
    if not fea_col_names:
        fea_col_names = list(test_df.columns)
    for name, group in test_df.groupby(grain_col_names):
        lr = lr_models[name]
        cur_pred = lr.predict(group[fea_col_names])
        dict1 = {
            time_col_name: list(group[time_col_name]),
            "prediction": cur_pred,
        }
        dict2 = dict(zip(grain_col_names, name))
        for grain in grain_col_names:
            dict2[grain] = [dict2[grain]] * len(cur_pred)
        cur_pred_df = pd.DataFrame({**dict1, **dict2})
        pred_dfs.append(cur_pred_df)
    pred_all = pd.concat(pred_dfs)
    pred_all.reset_index(drop=True, inplace=True)
    if nonnegative_output:
        pred_all["prediction"] = pred_all["prediction"].apply(lambda x: max(0, x))
    if integer_output:
        pred_all["prediction"] = pred_all["prediction"].apply(lambda x: round(x))
    return pred_all
