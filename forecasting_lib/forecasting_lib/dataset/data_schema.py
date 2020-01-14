# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd


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
