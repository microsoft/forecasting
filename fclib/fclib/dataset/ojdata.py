# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import subprocess
import pandas as pd
import math
import datetime
import itertools
import argparse
import logging
import requests
from tqdm import tqdm

from fclib.common.utils import git_repo_path
from fclib.feature_engineering.feature_utils import df_from_cartesian_product

DATA_FILE_LIST = ["yx.csv", "storedemo.csv"]
SCRIPT_NAME = "load_oj_data.R"

DEFAULT_TARGET_COL = "move"
DEFAULT_STATIC_FEA = None
DEFAULT_DYNAMIC_FEA = ["deal", "feat"]

# The start datetime of the first week in the record
FIRST_WEEK_START = pd.to_datetime("1989-09-14 00:00:00")

# Original data source
OJ_URL = "https://github.com/cran/bayesm/raw/master/data/orangeJuice.rda"


log = logging.getLogger(__name__)


def maybe_download(url, dest_directory, filename=None):
    """Download a file if it is not already downloaded.
    Args:
        dest_directory (str): Destination directory.
        url (str): URL of the file to download.
        filename (str): File name.
        
    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(dest_directory, exist_ok=True)
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(r.iter_content(block_size), total=num_iterables, unit="KB", unit_scale=True,):
                file.write(data)
    else:
        log.debug("File {} already downloaded".format(filepath))

    return filepath


def download_ojdata(dest_dir="."):
    """Download orange juice dataset from the original source.

     Args:
        dest_dir (str): Directory path for the downloaded file
    
    Returns:
        str: Path of the downloaded file.
    """
    url = OJ_URL
    rda_path = maybe_download(url, dest_directory=dest_dir)

    # Check if data files exist
    data_exists = True
    for f in DATA_FILE_LIST:
        file_path = os.path.join(dest_dir, f)
        data_exists = data_exists and os.path.exists(file_path)

    if not data_exists:
        # Call data loading script
        repo_path = git_repo_path()
        script_path = os.path.join(repo_path, "fclib", "fclib", "dataset", SCRIPT_NAME)

        try:
            print(f"Destination directory: {dest_dir}")
            output = subprocess.run(
                ["Rscript", script_path, rda_path, dest_dir], stderr=subprocess.PIPE, stdout=subprocess.PIPE
            )
            print(output.stdout)
            if output.returncode != 0:
                raise Exception(f"Subprocess failed - {output.stderr}")

        except subprocess.CalledProcessError as e:
            raise e
    else:
        print("Data already exists at the specified location.")


def complete_and_fill_df(df, stores, brands, weeks):
    """Completes missing rows in Orange Juice datasets and fills in the missing values.
    
    Args:
        df (pd.DataFrame): data frame to fill in the rows and missing values in
        stores (list[int]): list of stores to include
        brands (list[int]): list of brands to include
        weeks (list[int]): list of weeks to include
        
    Returns:
        pd.DataFrame: data frame with completed rows and missing values filled in
    
    """
    d = {"store": stores, "brand": brands, "week": weeks}
    data_grid = df_from_cartesian_product(d)
    # Complete all rows
    df_filled = pd.merge(data_grid, df, how="left", on=["store", "brand", "week"])
    # Fill in missing values
    df_filled = df_filled.groupby(["store", "brand"]).apply(lambda x: x.fillna(method="ffill").fillna(method="bfill"))

    return df_filled


def _gen_split_indices(n_splits=12, horizon=2, gap=2, first_week=40, last_week=156):
    """Generate week splits for given parameters"""
    test_start_index = last_week - (horizon * n_splits) + 1
    train_end_index_first = test_start_index - gap
    train_end_index_last = train_end_index_first + (n_splits - 1) * horizon

    assert (
        test_start_index >= first_week
    ), f"Please adjust your parameters, so that testing data (currently week {test_start_index}), \
         starts after the first available week (week {first_week})."

    assert (
        train_end_index_first >= first_week
    ), f"Please adjust your parameters, so that last training data point (currently week {train_end_index_first}) \
        comes after the first available week (week {first_week})."

    test_start_week_list = list(range(test_start_index, (last_week - horizon + 1) + 1, horizon))
    test_end_week_list = list(range(test_start_index + horizon - 1, last_week + 1, horizon))
    train_end_week_list = list(range(train_end_index_first, train_end_index_last + 1, horizon))
    return test_start_week_list, test_end_week_list, train_end_week_list


def split_train_test(data_dir, n_splits=1, horizon=2, gap=2, first_week=40, last_week=156, write_csv=False):
    """Generate training, testing, and auxiliary datasets. Training data includes the historical 
    sales and external features; testing data contains the future sales and external features; 
    auxiliary data includes the future price, deal, and advertisement information which can be 
    used for making predictions (we assume such auxiliary information is available at the time 
    when we generate the forecasts). Use this function to generate the train, test, aux data for
    each forecast period on the fly, or use write_csv flag to write data to files.

    Note that train_*.csv files in /train folder contain all the features in the training period
    and aux_*.csv files in /train folder contain all the features except 'logmove', 'constant',
    'profit' up until the forecast period end week. Both train_*.csv and auxi_*csv can be used for
    generating forecasts in each split. However, test_*.csv files in /test folder can only be used
    for model performance evaluation.

    Example:
        data_dir = "/home/ojdata"

        train, test, aux = split_train_test(data_dir=data_dir, n_splits=5, horizon=3, write_csv=True)

        print(len(train))
        print(len(test))
        print(len(aux))

    Args:
        data_dir (str): location of the download directory
        n_splits (int, optional): number of splits (folds) to generate (default: 1) 
        horizon (int, optional): forecasting horizon, number of weeks to forecast (default: 2) 
        gap (int, optional): gap between training and testing, number of weeks between last training 
            week and first test week (default: 2) 
        first_week (int, optional): first available week (default: 40) 
        last_week (int, optional): last available week (default: 156)
        write_csv (Boolean, optional): Whether to write out the data files or not (default: False)
    
    Returns:
        list[pandas.DataFrame]: a list containing train data frames for each split
        list[pandas.DataFrame]: a list containing test data frames for each split
        list[pandas.DataFrame]: a list containing aux data frames for each split
        
    """
    # Read sales data into dataframe
    sales = pd.read_csv(os.path.join(data_dir, "yx.csv"), index_col=0)

    if write_csv:
        TRAIN_DATA_DIR = os.path.join(data_dir, "train")
        TEST_DATA_DIR = os.path.join(data_dir, "test")
        if not os.path.isdir(TRAIN_DATA_DIR):
            os.mkdir(TRAIN_DATA_DIR)
        if not os.path.isdir(TEST_DATA_DIR):
            os.mkdir(TEST_DATA_DIR)

    train_df_list = list()
    test_df_list = list()
    aux_df_list = list()

    test_start_week_list, test_end_week_list, train_end_week_list = _gen_split_indices(
        n_splits, horizon, gap, first_week, last_week
    )

    for i in range(n_splits):
        data_mask = (sales.week >= first_week) & (sales.week <= train_end_week_list[i])
        train_df = sales[data_mask].copy()
        data_mask = (sales.week >= test_start_week_list[i]) & (sales.week <= test_end_week_list[i])
        test_df = sales[data_mask].copy()
        data_mask = (sales.week >= first_week) & (sales.week <= test_end_week_list[i])
        aux_df = sales[data_mask].copy()
        aux_df.drop(["logmove", "constant", "profit"], axis=1, inplace=True)

        if write_csv:
            roundstr = "_" + str(i + 1) if n_splits > 1 else ""
            train_df.to_csv(os.path.join(TRAIN_DATA_DIR, "train" + roundstr + ".csv"))
            test_df.to_csv(os.path.join(TEST_DATA_DIR, "test" + roundstr + ".csv"))
            aux_df.to_csv(os.path.join(TRAIN_DATA_DIR, "auxi" + roundstr + ".csv"))

        train_df_list.append(train_df)
        test_df_list.append(test_df)
        aux_df_list.append(aux_df)

    return train_df_list, test_df_list, aux_df_list


def specify_data_schema(
    df,
    time_col_name,
    target_col_name,
    frequency,
    time_format,
    ts_id_col_names=None,
    static_feat_names=None,
    dynamic_feat_names=None,
    description=None,
):
    """Specify the schema of a time series dataset.

        Args:
            df (Pandas DataFrame): input time series dataframe
            time_col_name (str): name of the timestamp column
            target_col_name (str): name of the target column that need to be forecasted
            frequency (str): frequency of the timestamps represented by the time series offset
                             aliases used in Pandas (e.g. "W" for weekly frequency). Please see
                             https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases 
                             for details.
            time_format (str): format of the timestamps (e.g., "%d.%m.%Y %H:%M:%S")
            ts_id_col_names (list): names of the columns for identifying a unique time series of
                                 the target variable
            static_feat_names (list): names of the feature columns that do not change over time
            dynamic_feat_names (list): names of the feature columns that can change over time
            description (str): description of the data (e.g., "training set", "testing set")

            Note that static_feat_names should include column names of the static features 
            other than those in ts_id_col_names. In addition, dynamic_feat_names should not 
            include the timestamp column and the target column. 

        Returns:
            df_config (dict): configuration of the time series data 
        
        TODO: Check if this is used before release.
        
        Examples:
            >>> # Case 1
            >>> sales = {"timestamp": ["01/01/2001", "03/01/2001", "02/01/2001"], 
            >>>          "sales": [1234, 2345, 1324],  
            >>>          "store": ["1001", "1002", "1001"], 
            >>>          "brand": ["1", "2", "1"], 
            >>>          "income": [53000, 65000, 53000], 
            >>>          "price": [10, 12, 11]}
            >>> df = pd.DataFrame(sales)
            >>> time_col_name = "timestamp"
            >>> target_col_name = "sales"
            >>> ts_id_col_names = ["store", "brand"]
            >>> static_feat_names = ["income"]
            >>> dynamic_feat_names = ["price"]
            >>> frequency = "MS" #monthly start
            >>> time_format = "%m/%d/%Y"
            >>> df_config = specify_data_schema(df, time_col_name,
            >>>                                 target_col_name, frequency,
            >>>                                 time_format, ts_id_col_names,
            >>>                                 static_feat_names, dynamic_feat_names)
            >>> print(df_config)
            {'time_col_name': 'timestamp', 'target_col_name': 'sales', 'frequency': 'MS', 'time_format': '%m/%d/%Y', 'ts_id_col_names': ['store', 'brand'], 'static_feat_names': ['income'], 'dynamic_feat_names': ['price'], 'description': None}

            >>> # Case 2
            >>> sales = {"timestamp": ["01/01/2001", "02/01/2001", "03/01/2001"], 
            >>>          "sales": [1234, 2345, 1324],  
            >>>          "store": ["1001", "1001", "1001"], 
            >>>          "brand": ["1", "1", "1"], 
            >>>          "income": [53000, 53000, 53000], 
            >>>          "price": [10, 12, 11]}
            >>> df = pd.DataFrame(sales)
            >>> time_col_name = "timestamp"
            >>> target_col_name = "sales"
            >>> ts_id_col_names = None
            >>> static_feat_names = ["store", "brand", "income"]
            >>> dynamic_feat_names = ["price"]
            >>> frequency = "MS" #monthly start
            >>> time_format = "%m/%d/%Y"
            >>> df_config = specify_data_schema(df, time_col_name,
            >>>                                 target_col_name, frequency,
            >>>                                 time_format, ts_id_col_names,
            >>>                                 static_feat_names, dynamic_feat_names)
            >>> print(df_config)
            {'time_col_name': 'timestamp', 'target_col_name': 'sales', 'frequency': 'MS', 'time_format': '%m/%d/%Y', 'ts_id_col_names': None, 'static_feat_names': ['store', 'brand', 'income'], 'dynamic_feat_names': ['price'], 'description': None}          
        """
    if len(df) == 0:
        raise ValueError("Input time series dataframe should not be empty.")

    df_col_names = list(df)
    _check_col_names(df_col_names, time_col_name, "timestamp")
    _check_col_names(df_col_names, target_col_name, "target")
    _check_time_format(df, time_col_name, time_format)
    _check_frequency(df, time_col_name, frequency, time_format, ts_id_col_names)
    if ts_id_col_names is not None:
        _check_col_names(df_col_names, ts_id_col_names, "name_list")
    if static_feat_names is not None:
        _check_col_names(df_col_names, static_feat_names, "name_list")
        _check_static_feat(df, ts_id_col_names, static_feat_names)
    if dynamic_feat_names is not None:
        _check_col_names(df_col_names, dynamic_feat_names, "name_list")

    # Configuration of the time series data
    df_config = {
        "time_col_name": time_col_name,
        "target_col_name": target_col_name,
        "frequency": frequency,
        "time_format": time_format,
        "ts_id_col_names": ts_id_col_names,
        "static_feat_names": static_feat_names,
        "dynamic_feat_names": dynamic_feat_names,
        "description": description,
    }
    return df_config


def _check_col_names(df_col_names, input_col_names, input_type):
    """Check if input column/feature names are valid.
    """
    if input_type in ["timestamp", "target"]:
        assert isinstance(input_col_names, str)
        if input_col_names not in df_col_names:
            raise ValueError("Invalid {} column name. It cannot be found in the input dataframe.".format(input_type))
    else:
        assert isinstance(input_col_names, list)
        for c in input_col_names:
            if c not in df_col_names:
                raise ValueError(c + " is an invalid column name. It cannot be found in the input dataframe.")


def _check_time_format(df, time_col_name, time_format):
    """Check if the timestamp format is valid.
    """
    try:
        pd.to_datetime(df[time_col_name], format=time_format)
    except Exception:
        raise ValueError("Incorrect date format is specified.")


def _check_frequency(df, time_col_name, frequency, time_format, ts_id_col_names):
    """Check if the data frequency is valid.
    """
    try:
        df[time_col_name] = pd.to_datetime(df[time_col_name], format=time_format)
        timestamps_all = pd.date_range(min(df[time_col_name]), end=max(df[time_col_name]), freq=frequency)
    except Exception:
        raise ValueError(
            "Input data frequency is invalid. Please use the aliases in "
            + "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases"
        )

    condition1 = (ts_id_col_names is None) and (not set(df[time_col_name]) <= set(timestamps_all))
    condition2 = (ts_id_col_names is not None) and (
        not all(df.groupby(ts_id_col_names).apply(lambda x: set(x[time_col_name]) <= set(timestamps_all)))
    )
    if condition1 or condition2:
        raise ValueError(
            "Timestamp(s) with irregular frequency in the input dataframe. Please make sure the frequency "
            + "of each time series is as what specified by 'frequency'."
        )


def _check_static_feat(df, ts_id_col_names, static_feat_names):
    """Check if the input static features change over time and include ts_id_col_names.
    """
    for feat in static_feat_names:
        condition1 = (ts_id_col_names is None) and (df[feat].nunique() > 1)
        condition2 = (ts_id_col_names is not None) and (df.groupby(ts_id_col_names)[feat].nunique().max() > 1)
        if condition1 or condition2:
            raise ValueError("Input feature column {} is supposed to be static but it is not.".format(feat))


def specify_retail_data_schema(
    data_dir,
    sales=None,
    target_col_name=DEFAULT_TARGET_COL,
    static_feat_names=DEFAULT_STATIC_FEA,
    dynamic_feat_names=DEFAULT_DYNAMIC_FEA,
    description=None,
):
    """Specify data schema of OrangeJuice dataset.

    Example:
        data_dir = "/home/forecasting/ojdata"
        df_config, sales = specify_retail_data_schema(data_dir)
        print(df_config)

    Args:
        sales (Pandas DataFrame): sales data in the current forecast split
        target_col_name (str): name of the target column that need to be forecasted
        static_feat_names (list): names of the feature columns that do not change over time
        dynamic_feat_names (list): names of the feature columns that can change over time
        description (str): description of the data (e.g., "training set", "testing set")

    Returns:
        df_config (dict): configuration of the time series data 
        df (Pandas DataFrame): sales data combined with store demographic features
    """
    # Read the 1st split of training data if "sales" is not specified
    if sales is None:
        print("Sales dataframe is not given! The 1st split of training data will be used.")
        sales = pd.read_csv(os.path.join(data_dir, "train", "train_round_1.csv"), index_col=False)
        aux = pd.read_csv(os.path.join(data_dir, "train", "aux_round_1.csv"), index_col=False)
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
    storedemo = pd.read_csv(os.path.join(data_dir, "storedemo.csv"), index_col=False)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Data download directory")
    args = parser.parse_args()

    download_ojdata(args.data_dir)
    # train, test, aux = split_train_test(data_dir=data_dir, n_splits=1, horizon=2, write_csv=True)

    # print((test[0].week))
    # print((test[1].week))
    # print((test[2].week))
    # print((test[3].week))
    # print((test[4].week))
