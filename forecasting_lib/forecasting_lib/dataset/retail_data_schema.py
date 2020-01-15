# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import datetime
import itertools
import pandas as pd
from forecasting_lib.dataset.data_schema import specify_data_schema
from forecasting_lib.dataset.retail.benchmark_paths import DATA_DIR
from forecasting_lib.dataset.retail.benchmark_settings import FIRST_WEEK_START

DEFAULT_TARGET_COL = "move"
DEFAULT_STATIC_FEA = None
DEFAULT_DYNAMIC_FEA = ["deal", "feat"]


def specify_retail_data_schema(
    sales=None,
    target_col_name=DEFAULT_TARGET_COL,
    static_feat_names=DEFAULT_STATIC_FEA,
    dynamic_feat_names=DEFAULT_DYNAMIC_FEA,
    description=None,
):
    """Specify data schema of OrangeJuice dataset.

    Args:
        sales (Pandas DataFrame): sales data in the current forecast round
        target_col_name (str): name of the target column that need to be forecasted
        static_feat_names (list): names of the feature columns that do not change over time
        dynamic_feat_names (list): names of the feature columns that can change over time
        description (str): description of the data (e.g., "training set", "testing set")

    Returns:
        df_config (dict): configuration of the time series data 
        df (Pandas DataFrame): sales data combined with store demographic features
    """
    # Read the 1st round training data if "sales" is not specified
    if sales is None:
        print("Sales dataframe is not given! The 1st round training data will be used.")
        sales = pd.read_csv(os.path.join(DATA_DIR, "train", "train_round_1.csv"), index_col=False)
        aux = pd.read_csv(os.path.join(DATA_DIR, "train", "aux_round_1.csv"), index_col=False)
        # Merge with future price, deal, and advertisement info
        aux_features = [
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
            "deal",
            "feat",
        ]
        sales = pd.merge(sales, aux, how="right", on=["store", "brand", "week"] + aux_features)

    # Read store demographic data
    storedemo = pd.read_csv(os.path.join(DATA_DIR, "storedemo.csv"), index_col=False)

    # Compute unit sales
    sales["move"] = sales["logmove"].apply(lambda x: round(math.exp(x)) if x > 0 else 0)

    # Make sure each time series has the same time span
    store_list = sales["store"].unique()
    brand_list = sales["brand"].unique()
    week_list = range(sales["week"].min(), sales["week"].max() + 1)
    item_list = list(itertools.product(store_list, brand_list, week_list))
    item_df = pd.DataFrame.from_records(item_list, columns=["store", "brand", "week"])
    sales = item_df.merge(sales, how="left", on=["store", "brand", "week"])

    # Merge with storedemo
    df = sales.merge(storedemo, how="left", left_on="store", right_on="STORE")
    df.drop("STORE", axis=1, inplace=True)

    # Create timestamp
    df["timestamp"] = df["week"].apply(lambda x: FIRST_WEEK_START + datetime.timedelta(days=(x - 1) * 7))

    df_config = specify_data_schema(
        df,
        time_col_name="timestamp",
        target_col_name=target_col_name,
        frequency="W-THU",
        time_format="%Y-%m-%d",
        ts_id_col_names=["store", "brand"],
        static_feat_names=static_feat_names,
        dynamic_feat_names=dynamic_feat_names,
        description=description,
    )
    return df_config, df


if __name__ == "__main__":
    df_config, sales = specify_retail_data_schema()
    print(df_config)
