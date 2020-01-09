# coding: utf-8

# Create input features for the boosted decision tree model.

import os
import sys
import math
import datetime
import pandas as pd

from sklearn.pipeline import Pipeline
from common.features.lag import LagFeaturizer
from common.features.rolling_window import RollingWindowFeaturizer
from common.features.stats import PopularityFeaturizer
from common.features.temporal import TemporalFeaturizer

# Append TSPerf path to sys.path
tsperf_dir = os.getcwd()
if tsperf_dir not in sys.path:
    sys.path.append(tsperf_dir)

# Import TSPerf components
from utils import df_from_cartesian_product
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings \
    as bs

pd.set_option("display.max_columns", None)


def oj_preprocess(
    df, aux_df, week_list, store_list, brand_list, train_df=None
):

    df["move"] = df["logmove"].apply(lambda x: round(math.exp(x)))
    df = df[["store", "brand", "week", "move"]].copy()

    # Create a dataframe to hold all necessary data
    d = {"store": store_list, "brand": brand_list, "week": week_list}
    data_grid = df_from_cartesian_product(d)
    data_filled = pd.merge(
        data_grid, df, how="left", on=["store", "brand", "week"]
    )

    # Get future price, deal, and advertisement info
    data_filled = pd.merge(
        data_filled, aux_df, how="left", on=["store", "brand", "week"]
    )

    # Fill missing values
    if train_df is not None:
        data_filled = pd.concat(train_df, data_filled)
        forecast_creation_time = train_df["week_start"].max()

    data_filled = data_filled.groupby(["store", "brand"]).apply(
        lambda x: x.fillna(method="ffill").fillna(method="bfill")
    )

    data_filled["week_start"] = data_filled["week"].apply(
        lambda x: bs.FIRST_WEEK_START + datetime.timedelta(days=(x - 1) * 7)
    )

    if train_df is not None:
        data_filled = data_filled.loc[
            data_filled["week_start"] > forecast_creation_time
        ].copy()

    return data_filled


def make_features(
    pred_round,
    train_dir,
    lags,
    window_size,
    offset,
    used_columns,
    store_list,
    brand_list,
):
    """Create a dataframe of the input features.

    Args:
        pred_round (Integer): Prediction round
        train_dir (String): Path of the training data directory
        lags (Numpy Array): Numpy array including all the lags
        window_size (Integer): Maximum step for computing the moving average
        offset (Integer): Length of training data skipped in the retraining
        used_columns (List): A list of names of columns used in model training
            (including target variable)
        store_list (Numpy Array): List of all the store IDs
        brand_list (Numpy Array): List of all the brand IDs

    Returns:
        features (Dataframe): Dataframe including all the input features and
            target variable
    """
    # Load training data
    train_df = pd.read_csv(
        os.path.join(train_dir, "train_round_" + str(pred_round + 1) + ".csv")
    )
    aux_df = pd.read_csv(
        os.path.join(train_dir, "aux_round_" + str(pred_round + 1) + ".csv")
    )
    week_list = range(
        bs.TRAIN_START_WEEK + offset, bs.TEST_END_WEEK_LIST[pred_round] + 1
    )

    train_df_preprocessed = oj_preprocess(
        train_df, aux_df, week_list, store_list, brand_list
    )

    df_config = {
        "time_col_name": "week_start",
        "ts_id_col_names": ["brand", "store"],
        "target_col_name": "move",
        "frequency": "W",
        "time_format": "%Y-%m-%d",
    }

    temporal_featurizer = TemporalFeaturizer(
        df_config=df_config, feature_list=["month_of_year", "week_of_month"]
    )

    popularity_featurizer = PopularityFeaturizer(
        df_config=df_config,
        id_col_name="brand",
        data_format="wide",
        feature_col_name="price",
        wide_col_names=[
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
        ],
        output_col_name="price_ratio",
        return_feature_col=True,
    )

    lag_featurizer = LagFeaturizer(
        df_config=df_config,
        input_col_names="move",
        lags=lags,
        future_value_available=True,
    )
    moving_average_featurizer = RollingWindowFeaturizer(
        df_config=df_config,
        input_col_names="move",
        window_size=window_size,
        window_args={"min_periods": 1, "center": False},
        future_value_available=True,
        rolling_gap=2,
    )

    feature_engineering_pipeline = Pipeline(
        [
            ("temporal", temporal_featurizer),
            ("popularity", popularity_featurizer),
            ("lag", lag_featurizer),
            ("moving_average", moving_average_featurizer),
        ]
    )

    features = feature_engineering_pipeline.transform(train_df_preprocessed)

    # Temporary code for result verification
    features.rename(
        mapper={
            "move_lag_2": "move_lag2",
            "move_lag_3": "move_lag3",
            "move_lag_4": "move_lag4",
            "move_lag_5": "move_lag5",
            "move_lag_6": "move_lag6",
            "move_lag_7": "move_lag7",
            "move_lag_8": "move_lag8",
            "move_lag_9": "move_lag9",
            "move_lag_10": "move_lag10",
            "move_lag_11": "move_lag11",
            "move_lag_12": "move_lag12",
            "move_lag_13": "move_lag13",
            "move_lag_14": "move_lag14",
            "move_lag_15": "move_lag15",
            "move_lag_16": "move_lag16",
            "move_lag_17": "move_lag17",
            "move_lag_18": "move_lag18",
            "move_lag_19": "move_lag19",
            "month_of_year": "month",
        },
        axis=1,
        inplace=True,
    )
    features = features[
        [
            "store",
            "brand",
            "week",
            "week_of_month",
            "month",
            "deal",
            "feat",
            "move",
            "price",
            "price_ratio",
            "move_lag2",
            "move_lag3",
            "move_lag4",
            "move_lag5",
            "move_lag6",
            "move_lag7",
            "move_lag8",
            "move_lag9",
            "move_lag10",
            "move_lag11",
            "move_lag12",
            "move_lag13",
            "move_lag14",
            "move_lag15",
            "move_lag16",
            "move_lag17",
            "move_lag18",
            "move_lag19",
            "move_mean",
        ]
    ]

    return features
