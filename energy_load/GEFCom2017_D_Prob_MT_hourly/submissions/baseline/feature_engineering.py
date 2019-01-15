"""
This script creates a set of commonly used features using the functions in
common.feature_utils, which serve as a set of baseline features.
Feel free to write your own feature engineering code to create new features by
calling the feature_utils functions with alternative parameters.
"""
import os, sys, getopt
from functools import reduce
from datetime import timedelta
import datetime
import pandas as pd
import numpy as np

import localpath
from energy_load.GEFCom2017_D_Prob_MT_hourly.common.benchmark_paths\
    import DATA_DIR, SUBMISSIONS_DIR

print('Data directory used: {}'.format(DATA_DIR))

TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_BASE_FILE = 'train_base.csv'
TRAIN_FILE_PREFIX = 'train_round_'
TEST_FILE_PREFIX = 'test_round_'
NUM_ROUND = 6

DATETIME_COLNAME = 'Datetime'
HOLIDAY_COLNAME = 'Holiday'

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


ALLOWED_TIME_COLUMN_TYPES = [pd.Timestamp, pd.DatetimeIndex,
                             datetime.datetime, datetime.date]


def is_datetime_like(x):
    """Function that checks if a data frame column x is of a datetime type."""
    return any(isinstance(x, col_type)
               for col_type in ALLOWED_TIME_COLUMN_TYPES)


def hour_of_day(datetime_col):
    """Returns the hour from a datetime variable."""
    return datetime_col.dt.hour


def month_of_year(date_time_col):
    """Returns the month from a datetime variable."""
    return date_time_col.dt.month


def fourier_approximation(t, n, period):
    """
    Generic helper function to create Fourier Series at different harmonies (n) and periods.

    Args:
        t: Datetime column.
        n: Harmonies, n=0, 1, 2, 3,...
        period: Period of the datetime variable t.
    
    Returns:
        x_sin: Sine component
        x_cos: Cosine component
    """
    x = n * 2 * np.pi * t/period
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
        output_dict: Output dictionary containing sine and cosine components of     the Fourier series for all harmonies.
    """
    day_of_year = datetime_col.dt.dayofyear

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(day_of_year, n, 365.24)

        output_dict['annual_sin_'+str(n)] = sin
        output_dict['annual_cos_'+str(n)] = cos

    return output_dict


def weekly_fourier(datetime_col, n_harmonics):
    """
    Creates Weekly Fourier Series at different harmonies (n).

    Args:
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...
    
    Returns:
        output_dict: Output dictionary containing sine and cosine components of     the Fourier series for all harmonies.
    """
    day_of_week = datetime_col.dt.dayofweek

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(day_of_week, n, 7)

        output_dict['weekly_sin_'+str(n)] = sin
        output_dict['weekly_cos_'+str(n)] = cos

    return output_dict


def daily_fourier(datetime_col, n_harmonics):
    """
    Creates Daily Fourier Series at different harmonies (n).

    Args:
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...
    
    Returns:
        output_dict: Output dictionary containing sine and cosine components of     the Fourier series for all harmonies.
    """
    hour_of_day = datetime_col.dt.hour + 1

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(hour_of_day, n, 24)

        output_dict['daily_sin_'+str(n)] = sin
        output_dict['daily_cos_'+str(n)] = cos

    return output_dict


def same_week_day_hour_lag(datetime_col, value_col, n_years=3,
                           week_window=1, agg_func='mean',
                           output_colname='SameWeekHourLag'):
    """
    This function creates a lag feature by averaging values of and around the same week, same day of week, and same hour of day, of previous years.
    
    Args:
        datetime_col: Datetime column.
        value_col: Feature value column to create lag feature from.
        n_years: Number of previous years data to use. Default value 3.
        week_window: Number of weeks before and after the same week to use,         which should help reduce noise in the data. Default value 1.
        agg_func: aggregation function to apply on multiple previous values.        Default value 'mean'.
        output_colname: name of the output lag feature column. Default value        'SameWeekHourLag'.

    Returns:
        df[[output_colname]]: pandas DataFrame containing the newly created lag     feature as a column.
    """

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)
    min_time_stamp = min(datetime_col)
    max_time_stamp = max(datetime_col)

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    week_lag_base = 52
    week_lag_last_year = list(range(week_lag_base - week_window,
                              week_lag_base + week_window + 1))
    week_lag_all = []
    for y in range(n_years):
        week_lag_all += [x + y * 52 for x in week_lag_last_year]

    week_lag_cols = []
    for w in week_lag_all:
        if (max_time_stamp - timedelta(weeks=w)) >= min_time_stamp:
            col_name = 'week_lag_' + str(w)
            week_lag_cols.append(col_name)

            lag_datetime = df.index.get_level_values(0) - timedelta(weeks=w)
            valid_lag_mask = lag_datetime >= min_time_stamp

            df[col_name] = np.nan

            df.loc[valid_lag_mask, col_name] = \
                df.loc[lag_datetime[valid_lag_mask], 'value'].values

    # Additional aggregation options will be added as needed
    if agg_func == 'mean':
        df[output_colname] = round(df[week_lag_cols].mean(axis=1))

    return df[[output_colname]]


def same_day_hour_lag(datetime_col, value_col, n_years=3,
                      day_window=1, agg_func='mean',
                      output_colname='SameDayHourLag'):
    """
    This function creates a lag feature by averaging values of and around the same day of year, and same hour of day, of previous years.
    
    Args:
        datetime_col: Datetime column.
        value_col: Feature value column to create lag feature from.
        n_years: Number of previous years data to use. Default value 3.
        day_window: Number of days before and after the same day to use, which      should help reduce noise in the data. Default value 1.
        agg_func: aggregation function to apply on multiple previous values.        Default value 'mean'.
        output_colname: name of the output lag feature column. Default value        'SameDayHourLag'.
    
    Returns:
        df[[output_colname]]: pandas DataFrame containing the newly created lag     feature as a column.
    """

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)
    min_time_stamp = min(datetime_col)
    max_time_stamp = max(datetime_col)

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    day_lag_base = 365
    day_lag_last_year = list(range(day_lag_base - day_window,
                                   day_lag_base + day_window + 1))
    day_lag_all = []
    for y in range(n_years):
        day_lag_all += [x + y * 365 for x in day_lag_last_year]

    day_lag_cols = []
    for d in day_lag_all:
        if (max_time_stamp - timedelta(days=d)) >= min_time_stamp:
            col_name = 'day_lag_' + str(d)
            day_lag_cols.append(col_name)

            lag_datetime = df.index.get_level_values(0) - timedelta(days=d)
            valid_lag_mask = lag_datetime >= min_time_stamp

            df[col_name] = np.nan

            df.loc[valid_lag_mask, col_name] = \
                df.loc[lag_datetime[valid_lag_mask], 'value'].values

    # Additional aggregation options will be added as needed
    if agg_func == 'mean':
        df[output_colname] = round(df[day_lag_cols].mean(axis=1))

    return df[[output_colname]]


def same_day_hour_moving_average(datetime_col, value_col, window_size,
                                 start_week, average_count, forecast_creation_time,
                                 output_col_prefix='moving_average_lag_'):
    """
    This function creates a moving average features by averaging values of the same day of week and same hour of day of previous weeks.

    Args:
        datetime_col: Datetime column
        value_col: Feature value column to create moving average features from.
        window_size: Number of weeks used to compute the average.
        start_week: First week of the first moving average feature.
        average_count: Number of moving average features to create.
        forecast_creation_time: The time point when the feature is created.         This value is used to prevent using data that are not available at      forecast creation time to compute features.
        output_col_prefix: Prefix of the output columns. The start week of each     moving average feature is added at the end. Default value               'moving_average_lag_'.

    Returns:
        df: pandas DataFrame containing the newly created lag features as           columns.

    For example, start_week = 9, window_size=4, and average_count = 3 will
    create three moving average features.
    1) moving_average_lag_9: average the same day and hour values of the 9th,
    10th, 11th, and 12th weeks before the current week.
    2) moving_average_lag_10: average the same day and hour values of the
    10th, 11th, 12th, and 13th weeks before the current week.
    3) moving_average_lag_11: average the same day and hour values of the
    11th, 12th, 13th, and 14th weeks before the current week.
    """

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    df = df.asfreq('H')

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    df['fct_diff'] = df.index - forecast_creation_time
    df['fct_diff'] = df['fct_diff'].apply(lambda x: x.days*24 + x.seconds/3600)
    max_diff = max(df['fct_diff'])

    for i in range(average_count):
        output_col = output_col_prefix + str(start_week+i)
        week_lag_start = start_week + i
        hour_lags = [(week_lag_start + w) * 24 * 7 for w in range(window_size)]
        hour_lags = [h for h in hour_lags if h > max_diff]
        if len(hour_lags) > 0:
            tmp_df = df[['value']].copy()
            tmp_col_all = []
            for h in hour_lags:
                tmp_col = 'tmp_lag_' + str(h)
                tmp_col_all.append(tmp_col)
                tmp_df[tmp_col] = tmp_df['value'].shift(h)

            df[output_col] = round(tmp_df[tmp_col_all].mean(axis=1))
    df.drop('value', inplace=True, axis=1)

    return df


def create_basic_features(input_df, datetime_colname):
    """
    This helper function uses the functions in common.feature_utils to
    create a set of basic features which are independently created for each
    row, i.e. no lag features or rolling window features.
    
    Args:
        input_df (pandas.DataFrame): data frame for which to compute basic features.
        datetime_colname (str): name of Datetime column

    Returns:
        output_df (pandas.DataFrame): output data frame which contains newly created features

    """

    output_df = input_df.copy()
    if not is_datetime_like(output_df[datetime_colname]):
        output_df[datetime_colname] = \
            pd.to_datetime(output_df[datetime_colname], format=DATETIME_FORMAT)
    datetime_col = output_df[datetime_colname]

    # Basic temporal features
    output_df['Hour'] = hour_of_day(datetime_col)
    output_df['MonthOfYear'] = month_of_year(datetime_col)

    # Fourier approximation features
    annual_fourier_approx = annual_fourier(datetime_col, n_harmonics=3)
    weekly_fourier_approx = weekly_fourier(datetime_col, n_harmonics=3)

    for k, v in annual_fourier_approx.items():
        output_df[k] = v

    for k, v in weekly_fourier_approx.items():
        output_df[k] = v

    return output_df


def create_advanced_features(train_df, test_df, datetime_colname,
                             holiday_colname=None):
    """
    This helper function uses the functions in common.feature_utils to
    create a set of advanced features. These features could depend on other
    rows in two ways:
    1) Lag or rolling window features depend on values of previous time points.
    2) Normalized features depend on the value range of the entire feature
    column.
    Therefore, the train_df and test_df are concatenated to create these
    features.
    
    NOTE: test_df can not contain any values that are unknown at
    forecasting creation time to avoid data leakage from the future. For
    example, it can contain the timestamps, zone, holiday, forecasted
    temperature, but it MUST NOT contain things like actual temperature,
    actual load, etc.

    Args:
        train_df (pandas.DataFrame): data frame containing training data
        test_df (pandas.DataFrame): data frame containing testing data
        datetime_colname (str): name of Datetime column
        holiday_colname (str): name of Holiday column (if present), default         value is None

    Returns:
        output_df_train (pandas.DataFrame): output containing newly constructed     features on training data
        output_df_test (pandas.DataFrame): output containing newly constructed      features on testing data
        
    """
    output_df = pd.concat([train_df, test_df], sort=True)
    if not is_datetime_like(output_df[datetime_colname]):
        output_df[datetime_colname] = \
            pd.to_datetime(output_df[datetime_colname], format=DATETIME_FORMAT)
    forecast_creation_time = max(train_df[datetime_colname])

    # Load lag feature based on previous years' load
    same_week_day_hour_load_lag = \
        output_df[[datetime_colname, 'DEMAND', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['DEMAND'],
                                             output_colname='LoadLag'))
    same_week_day_hour_load_lag.reset_index(inplace=True)

    same_day_hour_drybulb_lag = \
        output_df[[datetime_colname, 'DryBulb', 'Zone']].groupby('Zone').apply(
            lambda g: same_day_hour_lag(g[datetime_colname], g['DryBulb'],
                                        output_colname='DryBulbLag'))
    same_day_hour_drybulb_lag.reset_index(inplace=True)

    # Moving average features of load of recent weeks
    load_moving_average = \
        output_df[[datetime_colname, 'DEMAND', 'Zone']].groupby('Zone').apply(
            lambda g: same_day_hour_moving_average(g[datetime_colname],
                                                   g['DEMAND'],
                                                   start_week=10,
                                                   window_size=4,
                                                   average_count=7,
                                                   forecast_creation_time=forecast_creation_time,
                                                   output_col_prefix='RecentLoad_'))
    load_moving_average.reset_index(inplace=True)

    # Put everything together
    output_df = reduce(
        lambda left, right: pd.merge(left, right, on=[datetime_colname, 'Zone']),
        [output_df, same_week_day_hour_load_lag, same_day_hour_drybulb_lag,
         load_moving_average])

    same_week_day_hour_load_lag_10 = \
        output_df[[datetime_colname, 'RecentLoad_10', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['RecentLoad_10'],
                                             output_colname='LoadLag_10',
                                             n_years=5,
                                             week_window=0))
    same_week_day_hour_load_lag_10.reset_index(inplace=True)

    same_week_day_hour_load_lag_11 = \
        output_df[[datetime_colname, 'RecentLoad_11', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['RecentLoad_11'],
                                             output_colname='LoadLag_11',
                                             n_years=5,
                                             week_window=0))
    same_week_day_hour_load_lag_11.reset_index(inplace=True)

    same_week_day_hour_load_lag_12 = \
        output_df[[datetime_colname, 'RecentLoad_12', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['RecentLoad_12'],
                                             output_colname='LoadLag_12',
                                             n_years=5,
                                             week_window=0))
    same_week_day_hour_load_lag_12.reset_index(inplace=True)

    same_week_day_hour_load_lag_13 = \
        output_df[[datetime_colname, 'RecentLoad_13', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['RecentLoad_13'],
                                             output_colname='LoadLag_13',
                                             n_years=5,
                                             week_window=0))
    same_week_day_hour_load_lag_13.reset_index(inplace=True)

    same_week_day_hour_load_lag_14 = \
        output_df[[datetime_colname, 'RecentLoad_14', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['RecentLoad_14'],
                                             output_colname='LoadLag_14',
                                             n_years=5,
                                             week_window=0))
    same_week_day_hour_load_lag_14.reset_index(inplace=True)

    same_week_day_hour_load_lag_15 = \
        output_df[[datetime_colname, 'RecentLoad_15', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['RecentLoad_15'],
                                             output_colname='LoadLag_15',
                                             n_years=5,
                                             week_window=0))
    same_week_day_hour_load_lag_15.reset_index(inplace=True)

    same_week_day_hour_load_lag_16 = \
        output_df[[datetime_colname, 'RecentLoad_16', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['RecentLoad_16'],
                                             output_colname='LoadLag_16',
                                             n_years=5,
                                             week_window=0))
    same_week_day_hour_load_lag_16.reset_index(inplace=True)

    # Put everything together
    output_df = reduce(
        lambda left, right: pd.merge(left, right, on=[datetime_colname, 'Zone']),
        [output_df, same_week_day_hour_load_lag_10,
         same_week_day_hour_load_lag_11, same_week_day_hour_load_lag_12,
         same_week_day_hour_load_lag_13, same_week_day_hour_load_lag_14,
         same_week_day_hour_load_lag_15, same_week_day_hour_load_lag_16])

    columns_to_drop = []
    for i in range(10, 17):
        output_colname = 'LoadRatio_' + str(i)
        recent_colname = 'RecentLoad_' + str(i)
        lag_colname = 'LoadLag_' + str(i)
        columns_to_drop += [recent_colname, lag_colname]
        output_df[output_colname] = output_df[recent_colname]/output_df[lag_colname]

    output_df.drop(columns_to_drop, inplace=True, axis=1)

    # Split train and test data and return separately
    train_end = max(train_df[datetime_colname])
    output_df_train = output_df.loc[output_df[datetime_colname] <= train_end, ]
    output_df_test = output_df.loc[output_df[datetime_colname] > train_end, ]

    return output_df_train, output_df_test


def main(train_dir, test_dir, output_dir, datetime_colname, holiday_colname):
    """
    This helper function uses the create_basic_features and create_advanced
    features functions to create features for each train and test round.
    
    Args:
        train_dir (str): directory containing training data
        test_dir (str): directory containing testing data
        output_dir (str): directory to which to save the output files
        datetime_colname (str): name of Datetime column
        holiday_colname (str): name of Holiday column
    """

    output_train_dir = os.path.join(output_dir, 'train')
    output_test_dir = os.path.join(output_dir, 'test')
    if not os.path.isdir(output_train_dir):
        os.mkdir(output_train_dir)
    if not os.path.isdir(output_test_dir):
        os.mkdir(output_test_dir)

    train_base_df = pd.read_csv(os.path.join(train_dir, TRAIN_BASE_FILE),
                                parse_dates=[datetime_colname])

    # These features only need to be created once for all rounds
    train_base_basic_features = \
        create_basic_features(train_base_df,
                              datetime_colname=datetime_colname)

    for i in range(1, NUM_ROUND+1):
        train_file = os.path.join(train_dir, TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_file = os.path.join(test_dir, TEST_FILE_PREFIX + str(i) + '.csv')

        train_delta_df = pd.read_csv(train_file, parse_dates=[datetime_colname])
        test_round_df = pd.read_csv(test_file, parse_dates=[datetime_colname])

        train_delta_basic_features = \
            create_basic_features(train_delta_df,
                                  datetime_colname=datetime_colname)
        test_basic_features = \
            create_basic_features(test_round_df,
                                  datetime_colname=datetime_colname)

        train_round_df = pd.concat([train_base_df, train_delta_df])
        train_advanced_features, test_advanced_features = \
            create_advanced_features(train_round_df, test_round_df,
                                     datetime_colname=datetime_colname,
                                     holiday_colname=holiday_colname)

        train_basic_features = pd.concat([train_base_basic_features,
                                         train_delta_basic_features])

        # Drop some overlapping columns before merge basic and advanced
        # features.
        train_basic_columns = set(train_basic_features.columns)
        train_advanced_columns = set(train_advanced_features.columns)
        train_overlap_columns = list(train_basic_columns.
                                     intersection(train_advanced_columns))
        train_overlap_columns.remove('Zone')
        train_overlap_columns.remove('Datetime')
        train_advanced_features.drop(train_overlap_columns,
                                     inplace=True, axis=1)

        test_basic_columns = set(test_basic_features.columns)
        test_advanced_columns = set(test_advanced_features.columns)
        test_overlap_columns = list(test_basic_columns.
                                    intersection(test_advanced_columns))
        test_overlap_columns.remove('Zone')
        test_overlap_columns.remove('Datetime')
        test_advanced_features.drop(test_overlap_columns, inplace=True, axis=1)

        train_all_features = pd.merge(train_basic_features,
                                      train_advanced_features,
                                      on=['Zone', 'Datetime'])
        test_all_features = pd.merge(test_basic_features,
                                     test_advanced_features,
                                     on=['Zone', 'Datetime'])

        train_all_features.dropna(inplace=True, subset=['LoadLag', 'DryBulbLag'])
        test_all_features.drop(['DewPnt', 'DryBulb', 'DEMAND'],
                               inplace=True, axis=1)

        test_month = test_basic_features['MonthOfYear'].values[0]
        train_all_features = train_all_features.loc[
            train_all_features['MonthOfYear'] == test_month, ].copy()

        train_output_file = os.path.join(output_train_dir,
                                         TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_output_file = os.path.join(output_test_dir,
                                        TEST_FILE_PREFIX + str(i) + '.csv')

        train_all_features.to_csv(train_output_file, index=False)
        test_all_features.to_csv(test_output_file, index=False)

        print('Round {}'.format(i))
        print('Training data size: {}'.format(train_all_features.shape))
        print('Testing data size: {}'.format(test_all_features.shape))
        print('Minimum training timestamp: {}'.
              format(min(train_all_features[datetime_colname])))
        print('Maximum training timestamp: {}'.
              format(max(train_all_features[datetime_colname])))
        print('Minimum testing timestamp: {}'.
              format(min(test_all_features[datetime_colname])))
        print('Maximum testing timestamp: {}'.
              format(max(test_all_features[datetime_colname])))
        print('')


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], '', ['submission='])
    for opt, arg in opts:
        if opt == '--submission':
            submission_folder = arg
            output_data_dir = os.path.join(SUBMISSIONS_DIR, submission_folder, 'data')
            if not os.path.isdir(output_data_dir):
                os.mkdir(output_data_dir)
            OUTPUT_DIR = os.path.join(output_data_dir, 'features')
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    main(train_dir=TRAIN_DATA_DIR,
         test_dir=TEST_DATA_DIR,
         output_dir=OUTPUT_DIR,
         datetime_colname=DATETIME_COLNAME,
         holiday_colname=HOLIDAY_COLNAME)
