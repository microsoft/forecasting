# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta
from collections import Iterable

ALLOWED_TIME_COLUMN_TYPES = [
    pd.Timestamp,
    pd.DatetimeIndex,
    datetime.datetime,
    datetime.date,
]


def is_datetime_like(x):
    """Function that checks if a data frame column x is of a datetime type."""
    return any(isinstance(x, col_type) for col_type in ALLOWED_TIME_COLUMN_TYPES)


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


def split_train_validation(df, fct_horizon, datetime_colname):
    """
    Splits the input dataframe into train and validate folds based on the
    forecast creation time (fct) and forecast horizon specified by fct_horizon.

    Args:
        df: The input data frame to split.
        fct_horizon: list of tuples in the format of
            (fct, (forecast_horizon_start, forecast_horizon_end))
        datetime_colname: name of the datetime column

    Note: df[datetime_colname] needs to be a datetime type.
    """
    i_round = 0
    for fct, horizon in fct_horizon:
        i_round += 1
        train = df.loc[df[datetime_colname] < fct].copy()
        validation = df.loc[(df[datetime_colname] >= horizon[0]) & (df[datetime_colname] <= horizon[1]),].copy()

        yield i_round, train, validation


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
    if unit == "Y":
        new_datetime = input_datetime + relativedelta(years=add_count)
    elif unit == "M":
        new_datetime = input_datetime + relativedelta(months=add_count)
    elif unit == "W":
        new_datetime = input_datetime + relativedelta(weeks=add_count)
    elif unit == "D":
        new_datetime = input_datetime + relativedelta(days=add_count)
    elif unit == "h":
        new_datetime = input_datetime + relativedelta(hours=add_count)
    elif unit == "m":
        new_datetime = input_datetime + relativedelta(minutes=add_count)
    else:
        raise Exception(
            "Invalid backtest step unit, {}, provided. Valid " "step units are Y, M, W, D, h, " "and m".format(unit)
        )
    return new_datetime


def convert_to_tsdf(input_df, time_col_name, time_format):
    """
    Convert a time column in a data frame to monotonically increasing time
    index.
    Args:
        input_df(pandas.DataFrame): Input data frame to convert.
        time_col_name(str): Name of the time column to use as index.
        time_format(str): Format of the time column.

    Returns:
        pandas.DataFrame: A new data frame with the time column of the input
            data frame set as monotonically increasing index.
    """
    output_df = input_df.copy()
    if not is_datetime_like(output_df[time_col_name]):
        output_df[time_col_name] = pd.to_datetime(output_df[time_col_name], format=time_format)

    output_df.set_index(time_col_name, inplace=True)

    if not output_df.index.is_monotonic:
        output_df.sort_index(inplace=True)

    return output_df


def is_iterable_but_not_string(obj):
    """
    Determine if an object has iterable, list-like properties.
    Importantly, this functions *does not* consider a string
    to be list-like, even though Python strings are iterable.

    """
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def get_offset_by_frequency(frequency):
    frequency_to_offset_map = {
        "B": pd.offsets.BDay(),
        "C": pd.offsets.CDay(),
        "W": pd.offsets.Week(),
        "WOM": pd.offsets.WeekOfMonth(),
        "LWOM": pd.offsets.LastWeekOfMonth(),
        "M": pd.offsets.MonthEnd(),
        "MS": pd.offsets.MonthBegin(),
        "BM": pd.offsets.BMonthEnd(),
        "BMS": pd.offsets.BMonthBegin(),
        "CBM": pd.offsets.CBMonthEnd(),
        "CBMS": pd.offsets.CBMonthBegin(),
        "SM": pd.offsets.SemiMonthEnd(),
        "SMS": pd.offsets.SemiMonthBegin(),
        "Q": pd.offsets.QuarterEnd(),
        "QS": pd.offsets.QuarterBegin(),
        "BQ": pd.offsets.BQuarterEnd(),
        "BQS": pd.offsets.BQuarterBegin(),
        "REQ": pd.offsets.FY5253Quarter(),
        "A": pd.offsets.YearEnd(),
        "AS": pd.offsets.YearBegin(),
        "BYS": pd.offsets.YearBegin(),
        "BA": pd.offsets.BYearEnd(),
        "BAS": pd.offsets.BYearBegin(),
        "RE": pd.offsets.FY5253(),
        "BH": pd.offsets.BusinessHour(),
        "CBH": pd.offsets.CustomBusinessHour(),
        "D": pd.offsets.Day(),
        "H": pd.offsets.Hour(),
        "T": pd.offsets.Minute(),
        "min": pd.offsets.Minute(),
        "S": pd.offsets.Second(),
        "L": pd.offsets.Milli(),
        "ms": pd.offsets.Milli(),
        "U": pd.offsets.Micro(),
        "us": pd.offsets.Micro(),
        "N": pd.offsets.Nano(),
    }

    return frequency_to_offset_map[frequency]
