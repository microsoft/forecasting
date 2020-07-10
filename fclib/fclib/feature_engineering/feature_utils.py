# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License. 

"""
This file contains utility functions for creating features for time
series forecasting applications. All functions defined assume that
there is no missing data.
"""

import calendar
import itertools
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from dateutil.relativedelta import relativedelta

ALLOWED_TIME_COLUMN_TYPES = [
    pd.Timestamp,
    pd.DatetimeIndex,
    datetime.datetime,
    datetime.date,
]

# 0: Monday, 2: T/W/TR, 4: F, 5:SA, 6: S
WEEK_DAY_TYPE_MAP = {1: 2, 3: 2}  # Map for converting Wednesday and
# Thursday to have the same code as Tuesday
HOLIDAY_CODE = 7
SEMI_HOLIDAY_CODE = 8  # days before and after a holiday

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def is_datetime_like(x):
    """Function that checks if a data frame column x is of a datetime type."""
    return any(isinstance(x, col_type) for col_type in ALLOWED_TIME_COLUMN_TYPES)


def day_type(datetime_col, holiday_col=None, semi_holiday_offset=timedelta(days=1)):
    """
    Convert datetime_col to 7 day types
    0: Monday
    2: Tuesday, Wednesday, and Thursday
    4: Friday
    5: Saturday
    6: Sunday
    7: Holiday
    8: Days before and after a holiday

    Args:
        datetime_col: Datetime column.
        holiday_col: Holiday code column. Default value None.
        semi_holiday_offset: Time difference between the date before (or after)
            the holiday and the holiday. Default value timedelta(days=1).
    
    Returns:
        A numpy array containing converted datatime_col into day types.
    """

    datetype = pd.DataFrame({"DayType": datetime_col.dt.dayofweek})
    datetype.replace({"DayType": WEEK_DAY_TYPE_MAP}, inplace=True)

    if holiday_col is not None:
        holiday_mask = holiday_col > 0
        datetype.loc[holiday_mask, "DayType"] = HOLIDAY_CODE

        # Create a temporary Date column to calculate dates near the holidays
        datetype["Date"] = pd.to_datetime(datetime_col.dt.date, format=DATETIME_FORMAT)
        holiday_dates = set(datetype.loc[holiday_mask, "Date"])

        semi_holiday_dates = [
            pd.date_range(start=d - semi_holiday_offset, end=d + semi_holiday_offset, freq="D") for d in holiday_dates
        ]

        # Flatten the list of lists
        semi_holiday_dates = [d for dates in semi_holiday_dates for d in dates]

        semi_holiday_dates = set(semi_holiday_dates)
        semi_holiday_dates = semi_holiday_dates.difference(holiday_dates)

        datetype.loc[datetype["Date"].isin(semi_holiday_dates), "DayType"] = SEMI_HOLIDAY_CODE

    return datetype["DayType"].values


def hour_of_day(datetime_col):
    """Returns the hour from a datetime column."""
    return datetime_col.dt.hour


def time_of_year(datetime_col):
    """
    Time of year is a cyclic variable that indicates the annual position and
    repeats each year. It is each year linearly increasing over time going
    from 0 on January 1 at 00:00 to 1 on December 31st at 23:00. The values
    are normalized to be between [0; 1].

    Args:
        datetime_col: Datetime column.
    
    Returns:
        A numpy array containing converted datatime_col into time of year.
    """

    time_of_year = pd.DataFrame(
        {"DayOfYear": datetime_col.dt.dayofyear, "HourOfDay": datetime_col.dt.hour, "Year": datetime_col.dt.year}
    )
    time_of_year["TimeOfYear"] = (time_of_year["DayOfYear"] - 1) * 24 + time_of_year["HourOfDay"]

    time_of_year["YearLength"] = time_of_year["Year"].apply(lambda y: 366 if calendar.isleap(y) else 365)

    time_of_year["TimeOfYear"] = time_of_year["TimeOfYear"] / (time_of_year["YearLength"] * 24 - 1)

    return time_of_year["TimeOfYear"].values


def week_of_year(datetime_col):
    """Returns the week from a datetime column."""
    return datetime_col.dt.week


def week_of_month(date_time):
    """Returns the week of the month for a specified date.

    Args:
        dt (Datetime): Input date

    Returns:
        wom (Integer): Week of the month of the input date
    """

    def _week_of_month(date_time):
        from math import ceil

        first_day = date_time.replace(day=1)
        dom = date_time.day
        adjusted_dom = dom + first_day.weekday()
        wom = int(ceil(adjusted_dom / 7.0))
        return wom

    if isinstance(date_time, pd.Series):
        return date_time.apply(lambda x: _week_of_month(x))
    else:
        return _week_of_month(date_time)


def month_of_year(date_time_col):
    """Returns the month from a datetime column."""
    return date_time_col.dt.month


def day_of_week(date_time_col):
    """Returns the day of week from a datetime column."""
    return date_time_col.dt.dayofweek


def day_of_month(date_time_col):
    """Returns the day of month from a datetime column."""
    return date_time_col.dt.day


def day_of_year(date_time_col):
    """Returns the day of year from a datetime column."""
    return date_time_col.dt.dayofyear


# def encoded_month_of_year(month_of_year):
#     """
#     Create one hot encoding of month of year.
#     """
#     month_of_year = pd.get_dummies(month_of_year, prefix="MonthOfYear")

#     return month_of_year


# def encoded_day_of_week(day_of_week):
#     """
#     Create one hot encoding of day_of_week.
#     """
#     day_of_week = pd.get_dummies(day_of_week, prefix="DayOfWeek")

#     return day_of_week


# def encoded_day_of_month(day_of_month):
#     """
#     Create one hot encoding of day_of_month.
#     """
#     day_of_month = pd.get_dummies(day_of_month, prefix="DayOfMonth")

#     return day_of_month


# def encoded_day_of_year(day_of_year):
#     """
#     Create one hot encoding of day_of_year.
#     """
#     day_of_year = pd.get_dummies(day_of_year)

#     return day_of_year


# def encoded_hour_of_day(hour_of_day):
#     """
#     Create one hot encoding of hour_of_day.
#     """
#     hour_of_day = pd.get_dummies(hour_of_day, prefix="HourOfDay")

#     return hour_of_day


# def encoded_week_of_year(week_of_year):
#     """
#     Create one hot encoding of week_of_year.
#     """
#     week_of_year = pd.get_dummies(week_of_year, prefix="WeekOfYear")

#     return week_of_year


def normalized_current_year(datetime_col, min_year, max_year):
    """
    Temporal feature indicating the position of the year of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    
    Args:
        datetime_col: Datetime column.
        min_year: minimum value of year.
        max_year: maximum value of year.
    
    Returns:
        float: the position of the current year in the min_year:max_year range
    """
    year = datetime_col.dt.year

    if max_year != min_year:
        current_year = (year - min_year) / (max_year - min_year)
    else:
        current_year = pd.Series([0 for x in range(len(datetime_col))])

    return current_year


def normalized_current_date(datetime_col, min_date, max_date):
    """
    Temporal feature indicating the position of the date of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    
    Args:
        datetime_col: Datetime column.
        min_date: minimum value of date.
        max_date: maximum value of date.

    Returns:
        float: the position of the current date in the min_date:max_date range
    """
    date = datetime_col # .dt.date
    current_date = (date - min_date) # .apply(lambda x: x.days)

    if max_date != min_date:
        current_date = current_date / (max_date - min_date) # .days
    else:
        current_date = pd.Series([0 for x in range(len(datetime_col))])

    return current_date


def normalized_current_datehour(datetime_col, min_datehour, max_datehour):
    """
    Temporal feature indicating the position of the hour of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    
    Args:
        datetime_col: Datetime column.
        min_datehour: minimum value of datehour.
        max_datehour: maximum value of datehour.

    Returns:
        float: the position of the current datehour in the min_datehour:max_datehour range
    """
    current_datehour = (datetime_col - min_datehour).apply(lambda x: x.days * 24 + x.seconds / 3600)

    max_min_diff = max_datehour - min_datehour

    if max_min_diff != 0:
        current_datehour = current_datehour / (max_min_diff.days * 24 + max_min_diff.seconds / 3600)
    else:
        current_datehour = pd.Series([0 for x in range(len(datetime_col))])

    return current_datehour


def normalized_columns(datetime_col, value_col, mode="log", output_colname="normalized_columns"):
    """
    Creates columns normalized to be log of input columns devided by global average of each columns,
    or normalized using maximum and minimum.
    
    Args:
        datetime_col: Datetime column.
        value_col: Value column to be normalized.
        mode: Normalization mode,
            accepted values are 'log' and 'minmax'. Default value 'log'.
    
    Returns:
        Normalized value column.
    """

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)

    df = pd.DataFrame({"Datetime": datetime_col, "value": value_col})
    df.set_index("Datetime", inplace=True)

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    if mode == "log":
        mean_value = df["value"].mean()
        if mean_value != 0:
            df[output_colname] = np.log(df["value"] / mean_value)
        elif mean_value == 0:
            df[output_colname] = 0
    elif mode == "minmax":
        min_value = min(df["value"])
        max_value = max(df["value"])
        if min_value != max_value:
            df[output_colname] = (df["value"] - min_value) / (max_value - min_value)
        elif min_value == max_value:
            df[output_colname] = 0
    else:
        raise ValueError("Valid values for mode are 'log' and 'minmax'")

    return df[[output_colname]]


def fourier_approximation(t, n, period):
    """
    Generic helper function to create Fourier Series at different harmonies (n) and periods.

    Args:
        t: Datetime column.
        n: Harmonies, n=0, 1, 2, 3,...
        period: Period of the datetime variable t.
    
    Returns:
        float: Sine component
        float: Cosine component
    """
    x = n * 2 * np.pi * t / period
    x_sin = np.sin(x)
    x_cos = np.cos(x)

    return x_sin, x_cos


def annual_fourier(datetime_col, n_harmonics):
    """
    Creates Annual Fourier Series at different harmonies (n).

    Args:
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...
    
    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """
    day_of_year = datetime_col.dt.dayofyear

    output_dict = {}
    for n in range(1, n_harmonics + 1):
        sin, cos = fourier_approximation(day_of_year, n, 365.24)

        output_dict["annual_sin_" + str(n)] = sin
        output_dict["annual_cos_" + str(n)] = cos

    return output_dict


def weekly_fourier(datetime_col, n_harmonics):
    """
    Creates Weekly Fourier Series at different harmonies (n).

    Args:
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...
    
    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """
    day_of_week = datetime_col.dt.dayofweek + 1

    output_dict = {}
    for n in range(1, n_harmonics + 1):
        sin, cos = fourier_approximation(day_of_week, n, 7)

        output_dict["weekly_sin_" + str(n)] = sin
        output_dict["weekly_cos_" + str(n)] = cos

    return output_dict


def daily_fourier(datetime_col, n_harmonics):
    """
    Creates Daily Fourier Series at different harmonies (n).

    Args:
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...
    
    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """
    hour_of_day = datetime_col.dt.hour + 1

    output_dict = {}
    for n in range(1, n_harmonics + 1):
        sin, cos = fourier_approximation(hour_of_day, n, 24)

        output_dict["daily_sin_" + str(n)] = sin
        output_dict["daily_cos_" + str(n)] = cos

    return output_dict


def df_from_cartesian_product(dict_in):
    """Generate a Pandas dataframe from Cartesian product of lists.
    
    Args: 
        dict_in (Dictionary): Dictionary containing multiple lists, e.g. {"fea1": list1, "fea2": list2}
        
    Returns:
        df (Dataframe): Dataframe corresponding to the Caresian product of the lists
    """
    from itertools import product

    cart = list(product(*dict_in.values()))
    df = pd.DataFrame(cart, columns=dict_in.keys())
    return df


def lagged_features(df, lags):
    """Create lagged features based on time series data.
    
    Args:
        df (Dataframe): Input time series data sorted by time
        lags (List): Lag lengths
        
    Returns:
        fea (Dataframe): Lagged features 
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
        df (Dataframe): Input features as a dataframe
        start_step (Integer): Starting time step of rolling mean
        window_size (Integer): Windows size of rolling mean
    
    Returns:
        fea (Dataframe): Dataframe consisting of the moving averages
    """
    if window_size is None:
        # Use a large window to compute average over all historical data
        window_size = df.shape[0]
    fea = df.shift(start_step).rolling(min_periods=1, center=False, window=window_size).mean()
    fea.columns = fea.columns + "_mean"
    return fea


def combine_features(df, lag_fea, lags, window_size, used_columns):
    """Combine lag features, moving average features, and orignal features in the data.
    
    Args:
        df (Dataframe): Time series data including the target series and external features
        lag_fea (List): A list of column names for creating lagged features
        lags (Numpy Array): Numpy array including all the lags
        window_size (Integer): Window size of rolling mean
        used_columns (List): A list containing the names of columns that are needed in the 
        input dataframe (including the target column)
    
    Returns:
        fea_all (Dataframe): Dataframe including all the features
    """
    lagged_fea = lagged_features(df[lag_fea], lags)
    moving_avg = moving_averages(df[lag_fea], 2, window_size)
    fea_all = pd.concat([df[used_columns], lagged_fea, moving_avg], axis=1)
    return fea_all


def gen_sequence(df, seq_len, seq_cols, start_timestep=0, end_timestep=None):
    """Reshape time series features into an array of dimension (# of time steps, # of 
    features).  
    
    Args:
        df (pd.Dataframe): Dataframe including time series data for a specific grain of a 
            multi-granular time series, e.g., data of a specific store-brand combination for 
            time series data involving multiple stores and brands
        seq_len (int): Number of previous time series values to be used to form feature
        sequences which can be used for model training
        seq_cols (list[str]): A list of names of the feature columns 
        start_timestep (int): First time step you can use to create feature sequences
        end_timestep (int): Last time step you can use to create feature sequences
        
    Returns:
        object: A generator object for iterating all the feature sequences
    """
    data_array = df[seq_cols].values
    if end_timestep is None:
        end_timestep = df.shape[0]
    for start, stop in zip(
        range(start_timestep, end_timestep - seq_len + 2), range(start_timestep + seq_len, end_timestep + 2)
    ):
        yield data_array[start:stop, :]


def gen_sequence_array(df_all, seq_len, seq_cols, grain1_name, grain2_name, start_timestep=0, end_timestep=None):
    """Combine feature sequences for all the combinations of (grain1_name, grain2_name) into a 
    3-dimensional array.
    
    Args:
        df_all (pd.Dataframe): Time series data of all the grains for multi-granular data
        seq_len (int): Number of previous time series values to be used to form sequences
        seq_cols (list[str]): A list of names of the feature columns 
        grain1_name (str): Name of the 1st column indicating the time series graunularity
        grain2_name (str): Name of the 2nd column indicating the time series graunularity
        start_timestep (int): First time step you can use to create feature sequences
        end_timestep (int): Last time step you can use to create feature sequences
        
    Returns:
        seq_array (np.array): An array of feature sequences for all combinations of granularities
    """
    seq_gen = (
        list(
            gen_sequence(
                df_all[(df_all[grain1_name] == grain1) & (df_all[grain2_name] == grain2)],
                seq_len,
                seq_cols,
                start_timestep,
                end_timestep,
            )
        )
        for grain1, grain2 in itertools.product(df_all[grain1_name].unique(), df_all[grain2_name].unique())
    )
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    return seq_array


def static_feature_array(df_all, total_timesteps, seq_cols, grain1_name, grain2_name):
    """Generate an arary which encodes all the static features.
    
    Args:
        df_all (pd.DataFrame): Time series data of all the grains for multi-granular data
        total_timesteps (int): Total number of training samples for modeling
        seq_cols (list[str]): A list of names of the static feature columns, e.g. store ID
        grain1_name (str): Name of the 1st column indicating the time series graunularity
        grain2_name (str): Name of the 2nd column indicating the time series graunularity
        
    Return:
        fea_array (np.array): An array of static features of all the grains, e.g. all the
            combinations of stores and brands in retail sale forecasting
    """
    fea_df = (
        df_all.groupby([grain1_name, grain2_name]).apply(lambda x: x.iloc[:total_timesteps, :]).reset_index(drop=True)
    )
    fea_array = fea_df[seq_cols].values
    return fea_array


def normalize_columns(df, seq_cols, scaler=MinMaxScaler()):
    """Normalize a subset of columns of a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe 
        seq_cols (list[str]): A list of names of columns to be normalized
        scaler (object): A scikit learn scaler object
    
    Returns:
        pd.DataFrame: Normalized dataframe
        object: Scaler object
    """
    cols_fixed = df.columns.difference(seq_cols)
    df_scaled = pd.DataFrame(scaler.fit_transform(df[seq_cols]), columns=seq_cols, index=df.index)
    df_scaled = pd.concat([df[cols_fixed], df_scaled], axis=1)
    return df_scaled, scaler


def get_datetime_col(df, datetime_colname):
    """
    Helper function for extracting the datetime column as datetime type from
    a data frame.

    Args:
        df: pandas DataFrame containing the column to convert
        datetime_colname: name of the column to be converted

    Returns:
        pandas.Series: converted column

    Raises:
        Exception: if datetime_colname does not exist in the dateframe df.
        Exception: if datetime_colname cannot be converted to datetime type.
    """
    if datetime_colname in df.index.names:
        datetime_col = df.index.get_level_values(datetime_colname)
    elif datetime_colname in df.columns:
        datetime_col = df[datetime_colname]
    else:
        raise Exception("Column or index {0} does not exist in the data " "frame".format(datetime_colname))

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(df[datetime_colname])
    return datetime_col


def get_month_day_range(date):
    """
    Returns the first date and last date of the month of the given date.
    """
    # Replace the date in the original timestamp with day 1
    first_day = date + relativedelta(day=1)
    # Replace the date in the original timestamp with day 1
    # Add a month to get to the first day of the next month
    # Subtract one day to get the last day of the current month
    last_day = date + relativedelta(day=1, months=1, days=-1, hours=23)
    return first_day, last_day


def add_datetime(input_datetime, unit, add_count):
    """
    Function to add a specified units of time (years, months, weeks, days,
    hours, or minutes) to the input datetime.

    Args:
        input_datetime: datatime to be added to
        unit: unit of time, valid values: 'year', 'month', 'week',
            'day', 'hour', 'minute'.
        add_count: number of units to add

    Returns:
        New datetime after adding the time difference to input datetime.

    Raises:
        Exception: if invalid unit is provided. Valid units are:
            'year', 'month', 'week', 'day', 'hour', 'minute'.
    """
    if unit == "year":
        new_datetime = input_datetime + relativedelta(years=add_count)
    elif unit == "month":
        new_datetime = input_datetime + relativedelta(months=add_count)
    elif unit == "week":
        new_datetime = input_datetime + relativedelta(weeks=add_count)
    elif unit == "day":
        new_datetime = input_datetime + relativedelta(days=add_count)
    elif unit == "hour":
        new_datetime = input_datetime + relativedelta(hours=add_count)
    elif unit == "minute":
        new_datetime = input_datetime + relativedelta(minutes=add_count)
    else:
        raise Exception(
            "Invalid backtest step unit, {}, provided. Valid " "step units are Y, M, W, D, h, " "and m".format(unit)
        )
    return new_datetime
