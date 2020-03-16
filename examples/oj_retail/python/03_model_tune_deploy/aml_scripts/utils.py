# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Utility functions for preparing features for training the LightGBM model and making predictions with the model.
"""

import os
import math
import datetime
import pandas as pd

FIRST_WEEK = 40
GAP = 2
HORIZON = 2
FIRST_WEEK_START = pd.to_datetime("1989-09-14 00:00:00")


def week_of_month(dt):
    """Get the week of the month for the specified date.
    
    Args: 
        dt (datetime): Input date
        
    Returns:
        int: Week of the month of the input date
    """
    from math import ceil

    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    wom = int(ceil(adjusted_dom / 7.0))
    return wom


def df_from_cartesian_product(dict_in):
    """Generate a Pandas dataframe from Cartesian product of lists.
    
    Args: 
        dict_in (dict): Dictionary containing multiple lists
        
    Returns:
        pd.Dataframe: Dataframe corresponding to the Caresian product of the lists
    """
    from collections import OrderedDict
    from itertools import product

    od = OrderedDict(sorted(dict_in.items()))
    cart = list(product(*od.values()))
    df = pd.DataFrame(cart, columns=od.keys())
    return df


def lagged_features(df, lags):
    """Create lagged features based on time series data.
    
    Args:
        df (pd.Dataframe): Input time series data sorted by time
        lags (list): Lag lengths
        
    Returns:
        pd.Dataframe: Lagged features 
    """
    df_list = []
    for lag in lags:
        df_shifted = df.shift(lag)
        df_shifted.columns = [x + "_lag" + str(lag) for x in df_shifted.columns]
        df_list.append(df_shifted)
    fea = pd.concat(df_list, axis=1)
    return fea


def moving_averages(df, start_step, window_size=None):
    """Compute averages of every feature over moving time windows.
    
    Args:
        df (pd.Dataframe): Input features as a dataframe
        start_step (int): Starting time step of rolling mean
        window_size (int): Windows size of rolling mean
    
    Returns:
        pd.Dataframe: Dataframe consisting of the moving averages
    """
    if window_size is None:  # Use a large window to compute average over all historical data
        window_size = df.shape[0]
    fea = df.shift(start_step).rolling(min_periods=1, center=False, window=window_size).mean()
    fea.columns = fea.columns + "_mean"
    return fea


def combine_features(df, lag_fea, lags, window_size, used_columns):
    """Combine different features for a certain store-brand.
    
    Args:
        df (pd.Dataframe): Time series data of a certain store-brand
        lag_fea (list): A list of column names for creating lagged features
        lags (np.array): Numpy array including all the lags
        window_size (int): Windows size of rolling mean
        used_columns (list): A list of names of columns used in model training (including target variable)
    
    Returns:
        pd.Dataframe: Dataframe including all features for the specific store-brand
    """
    lagged_fea = lagged_features(df[lag_fea], lags)
    moving_avg = moving_averages(df[lag_fea], 2, window_size)
    fea_all = pd.concat([df[used_columns], lagged_fea, moving_avg], axis=1)
    return fea_all


def create_features(pred_round, train_dir, lags, window_size, used_columns):
    """Create input features for model training and testing.

    Args: 
        pred_round (int): Prediction round (1, 2, ...)
        train_dir (str): Path of the training data directory 
        lags (np.array): Numpy array including all the lags
        window_size (int): Maximum step for computing the moving average
        used_columns (list[str]): A list of names of columns used in model training (including target variable)

    Returns:
        pd.Dataframe: Dataframe including all the input features and target variable
        int: Last week of the training data 
    """

    # Load training data
    default_train_file = os.path.join(train_dir, "train.csv")
    if os.path.isfile(default_train_file):
        train_df = pd.read_csv(default_train_file)
    else:
        train_df = pd.read_csv(os.path.join(train_dir, "train_" + str(pred_round) + ".csv"))
    train_df["move"] = train_df["logmove"].apply(lambda x: round(math.exp(x)))
    train_df = train_df[["store", "brand", "week", "move"]]

    # Create a dataframe to hold all necessary data
    store_list = train_df["store"].unique()
    brand_list = train_df["brand"].unique()
    train_end_week = train_df["week"].max()
    week_list = range(FIRST_WEEK, train_end_week + GAP + HORIZON)
    d = {"store": store_list, "brand": brand_list, "week": week_list}
    data_grid = df_from_cartesian_product(d)
    data_filled = pd.merge(data_grid, train_df, how="left", on=["store", "brand", "week"])

    # Get future price, deal, and advertisement info
    default_aux_file = os.path.join(train_dir, "auxi.csv")
    if os.path.isfile(default_aux_file):
        aux_df = pd.read_csv(default_aux_file)
    else:
        aux_df = pd.read_csv(os.path.join(train_dir, "auxi_" + str(pred_round) + ".csv"))
    data_filled = pd.merge(data_filled, aux_df, how="left", on=["store", "brand", "week"])

    # Create relative price feature
    price_cols = [
        "price1",
        "price2",
        "price3",
        "price4",
        "price5",
        "price6",
        "price7",
        "price8",
        "price9",
        "price10",
        "price11",
    ]
    data_filled["price"] = data_filled.apply(lambda x: x.loc["price" + str(int(x.loc["brand"]))], axis=1)
    data_filled["avg_price"] = data_filled[price_cols].sum(axis=1).apply(lambda x: x / len(price_cols))
    data_filled["price_ratio"] = data_filled["price"] / data_filled["avg_price"]
    data_filled.drop(price_cols, axis=1, inplace=True)

    # Fill missing values
    data_filled = data_filled.groupby(["store", "brand"]).apply(
        lambda x: x.fillna(method="ffill").fillna(method="bfill")
    )

    # Create datetime features
    data_filled["week_start"] = data_filled["week"].apply(
        lambda x: FIRST_WEEK_START + datetime.timedelta(days=(x - 1) * 7)
    )
    data_filled["year"] = data_filled["week_start"].apply(lambda x: x.year)
    data_filled["month"] = data_filled["week_start"].apply(lambda x: x.month)
    data_filled["week_of_month"] = data_filled["week_start"].apply(lambda x: week_of_month(x))
    data_filled["day"] = data_filled["week_start"].apply(lambda x: x.day)
    data_filled.drop("week_start", axis=1, inplace=True)

    # Create other features (lagged features, moving averages, etc.)
    features = data_filled.groupby(["store", "brand"]).apply(
        lambda x: combine_features(x, ["move"], lags, window_size, used_columns)
    )

    # Drop rows with NaN values
    features.dropna(inplace=True)

    return features, train_end_week
