# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd
from fclib.data_schema import specify_data_schema

DEFAULT_TARGET_COL = "DEMAND"
DEFAULT_STATIC_FEA = None
DEFAULT_DYNAMIC_FEA = ["DewPnt", "DryBulb", "Holiday"]

# TODO: resolve this
DATA_DIR = "TBD"


def specify_energy_data_schema(
    df=None,
    target_col_name=DEFAULT_TARGET_COL,
    static_feat_names=DEFAULT_STATIC_FEA,
    dynamic_feat_names=DEFAULT_DYNAMIC_FEA,
    description=None,
):
    """Specify data schema of GEFCom2017 dataset.

    Args:
        df (Pandas DataFrame): energy demand data in the current forecast round
        target_col_name (str): name of the target column that need to be forecasted
        static_feat_names (list): names of the feature columns that do not change over time
        dynamic_feat_names (list): names of the feature columns that can change over time
        description (str): description of the data (e.g., "training set", "testing set")

    Returns:
        df_config (dict): configuration of the time series data 
    """
    # Read the 1st round training data if "df" is not specified
    if df is None:
        print("Energy demand dataframe is not given! The 1st round training data will be used.")
        train_base = pd.read_csv(os.path.join(DATA_DIR, "train", "train_base.csv"), parse_dates=["Datetime"])
        train_round_1 = pd.read_csv(os.path.join(DATA_DIR, "train", "train_round_1.csv"), parse_dates=["Datetime"])
        df = pd.concat([train_base, train_round_1]).reset_index(drop=True)

    df_config = specify_data_schema(
        df,
        time_col_name="Datetime",
        target_col_name=target_col_name,
        frequency="H",
        time_format="%Y-%m-%d %H:%M:%S",
        ts_id_col_names=["Zone"],
        static_feat_names=static_feat_names,
        dynamic_feat_names=dynamic_feat_names,
        description=description,
    )
    return df_config


if __name__ == "__main__":
    df_config = specify_energy_data_schema()
    print(df_config)
