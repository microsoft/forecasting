# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import subprocess
import pandas as pd
import math
import datetime
import itertools

DATA_FILE_LIST = ["yx.csv", "storedemo.csv"]
SCRIPT_NAME = "download_oj_data.R"

DEFAULT_TARGET_COL = "move"
DEFAULT_STATIC_FEA = None
DEFAULT_DYNAMIC_FEA = ["deal", "feat"]


def download_ojdata(dest_dir):
    """Downloads Orange Juice dataset.

    Args:
        dest_dir (str): Directory path for the downloaded file
    """
    maybe_download(dest_dir=dest_dir)


def maybe_download(dest_dir):
    """Download a file if it is not already downloaded.
    
    Args:
        dest_dir (str): Destination directory
        
    Returns:
        str: File path of the file downloaded.
    """
    # Check if data files exist
    data_exists = True
    for f in DATA_FILE_LIST:
        file_path = os.path.join(dest_dir, f)
        data_exists = data_exists and os.path.exists(file_path)

    if not data_exists:
        # Call data download script
        print("Starting data download ...")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPT_NAME)
        try:
            subprocess.call(["Rscript", script_path, dest_dir])
        except subprocess.CalledProcessError as e:
            print(e.output)
    else:
        print("Data already exists at the specified location.")


def split_train_test(data_dir, forecast_settings, write_csv=False):
    """Generate training, testing, and auxiliary datasets. Training data includes the historical 
    sales and external features; testing data contains the future sales and external features; 
    auxiliary data includes the future price, deal, and advertisement information which can be 
    used for making predictions (we assume such auxiliary information is available at the time 
    when we generate the forecasts). Use this function to generate the train, test, aux data for
    each forecast period on the fly, or use write_csv flag to write data to files.

    Note that train_*.csv files in /train folder contain all the features in the training period
    and aux_*.csv files in /train folder contain all the features except 'logmove', 'constant',
    'profit' up until the forecast period end week. Both train_*.csv and aux_*csv can be used for
    generating forecasts in each round. However, test_*.csv files in /test folder can only be used
    for model performance evaluation.

    Example:
        from fclib.common.utils import forecast_settings

        data_dir = "/home/forecasting/ojdata"

        for train, test, aux in split_train_test(data_dir=data_dir, forecast_settings=forecast_settings):
            print("Training data size: {}".format(train.shape))
            print("Testing data size: {}".format(test.shape))
            print("Auxiliary data size: {}".format(aux.shape))
            print("Minimum training week number: {}".format(min(train["week"])))
            print("Maximum training week number: {}".format(max(train["week"])))
            print("Minimum testing week number: {}".format(min(test["week"])))
            print("Maximum testing week number: {}".format(max(test["week"])))
            print("Minimum auxiliary week number: {}".format(min(aux["week"])))
            print("Maximum auxiliary week number: {}".format(max(aux["week"])))
            print("")

    Args:
        data_dir (str): location of the download directory
        forecast_settings (dict): dictionary containing forecast experiment parameters
        write_csv (Boolean): Whether to write out the data files or not
        
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

    for i in range(forecast_settings.NUM_ROUNDS):
        data_mask = (sales.week >= forecast_settings.TRAIN_START_WEEK) & (
            sales.week <= forecast_settings.TRAIN_END_WEEK_LIST[i]
        )
        train = sales[data_mask].copy()
        data_mask = (sales.week >= forecast_settings.TEST_START_WEEK_LIST[i]) & (
            sales.week <= forecast_settings.TEST_END_WEEK_LIST[i]
        )
        test = sales[data_mask].copy()
        data_mask = (sales.week >= forecast_settings.TRAIN_START_WEEK) & (
            sales.week <= forecast_settings.TEST_END_WEEK_LIST[i]
        )
        aux = sales[data_mask].copy()
        aux.drop(["logmove", "constant", "profit"], axis=1, inplace=True)

        if write_csv:
            roundstr = "_" + str(i + 1) if forecast_settings.NUM_ROUNDS > 1 else ""
            train.to_csv(os.path.join(TRAIN_DATA_DIR, "train" + roundstr + ".csv"))
            test.to_csv(os.path.join(TEST_DATA_DIR, "test" + roundstr + ".csv"))
            aux.to_csv(os.path.join(TRAIN_DATA_DIR, "aux" + roundstr + ".csv"))
        yield train, test, aux


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
    forecast_settings,
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
    df["timestamp"] = df["week"].apply(
        lambda x: forecast_settings.FIRST_WEEK_START + datetime.timedelta(days=(x - 1) * 7)
    )

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
    from fclib.common import forecast_settings

    forecast_settings.NUM_ROUNDS = 3
    data_dir = "/home/vapaunic/forecasting/ojdata"

    for train, test, aux in split_train_test(data_dir=data_dir, forecast_settings=forecast_settings, write_csv=True):
        print("Training data size: {}".format(train.shape))
        print("Testing data size: {}".format(test.shape))
        print("Auxiliary data size: {}".format(aux.shape))
        print("Minimum training week number: {}".format(min(train["week"])))
        print("Maximum training week number: {}".format(max(train["week"])))
        print("Minimum testing week number: {}".format(min(test["week"])))
        print("Maximum testing week number: {}".format(max(test["week"])))
        print("Minimum auxiliary week number: {}".format(min(aux["week"])))
        print("Maximum auxiliary week number: {}".format(max(aux["week"])))
        print("")
