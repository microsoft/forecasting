# coding: utf-8

# Create input features for the Dilated Convolutional Neural Network (CNN) model.

import os
import sys
import math
import datetime
import numpy as np
import pandas as pd

# Append TSPerf path to sys.path
tsperf_dir = "."
if tsperf_dir not in sys.path:
    sys.path.append(tsperf_dir)

# Import TSPerf components
from utils import *
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs


def make_features(pred_round, train_dir, pred_steps, offset, store_list, brand_list):
    """Create a dataframe of the input features.
    
    Args: 
        pred_round (Integer): Prediction round
        train_dir (String): Path of the training data directory
        pred_steps (Integer): Number of prediction steps
        offset (Integer): Length of training data skipped in the retraining
        store_list (Numpy Array): List of all the store IDs
        brand_list (Numpy Array): List of all the brand IDs
        
    Returns:
        data_filled (Dataframe): Dataframe including the input features
        data_scaled (Dataframe): Dataframe including the normalized features 
    """
    # Load training data
    train_df = pd.read_csv(os.path.join(train_dir, "train_round_" + str(pred_round + 1) + ".csv"))
    train_df["move"] = train_df["logmove"].apply(lambda x: round(math.exp(x)))
    train_df = train_df[["store", "brand", "week", "move"]]

    # Create a dataframe to hold all necessary data
    week_list = range(bs.TRAIN_START_WEEK + offset, bs.TEST_END_WEEK_LIST[pred_round] + 1)
    d = {"store": store_list, "brand": brand_list, "week": week_list}
    data_grid = df_from_cartesian_product(d)
    data_filled = pd.merge(data_grid, train_df, how="left", on=["store", "brand", "week"])

    # Get future price, deal, and advertisement info
    aux_df = pd.read_csv(os.path.join(train_dir, "aux_round_" + str(pred_round + 1) + ".csv"))
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
        lambda x: bs.FIRST_WEEK_START + datetime.timedelta(days=(x - 1) * 7)
    )
    data_filled["month"] = data_filled["week_start"].apply(lambda x: x.month)
    data_filled["week_of_month"] = data_filled["week_start"].apply(lambda x: week_of_month(x))
    data_filled.drop("week_start", axis=1, inplace=True)

    # Normalize the dataframe of features
    cols_normalize = data_filled.columns.difference(["store", "brand", "week"])
    data_scaled, min_max_scaler = normalize_dataframe(data_filled, cols_normalize)

    return data_filled, data_scaled
