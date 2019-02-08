"""
This script creates a set of commonly used features using the functions in
common.feature_utils, which serve as a set of baseline features.
Feel free to write your own feature engineering code to create new features by
calling the feature_utils functions with alternative parameters.
"""

from .benchmark_paths import DATA_DIR
from common.feature_utils import *
from common.utils import is_datetime_like

import os
import pandas as pd
pd.set_option('display.max_columns', None)

print('Data directory used: {}'.format(DATA_DIR))

OUTPUT_DIR = os.path.join(DATA_DIR, 'features')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_BASE_FILE = 'train_base.csv'
TRAIN_FILE_PREFIX = 'train_round_'
TEST_FILE_PREFIX = 'test_round_'
NUM_ROUND = 6

DATETIME_COLNAME = 'Datetime'
HOLIDAY_COLNAME = 'Holiday'
GRAIN_COLNAME = 'Zone'

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# A dictionary mapping each feature name to the function for computing the
# feature
feature_function_dict = {'Hour': hour_of_day,
                         'DayOfWeek': day_of_week,
                         'DayOfMonth': day_of_month,
                         'TimeOfYear': time_of_year,
                         'WeekOfYear': week_of_year,
                         'MonthOfYear': month_of_year,
                         'AnnualFourier': annual_fourier,
                         'WeeklyFourier': weekly_fourier,
                         'DailyFourier': daily_fourier,
                         'CurrentDate': normalized_current_date,
                         'CurrentDateHour':
                             normalized_current_datehour,
                         'CurrentYear': normalized_current_year,
                         'DayType': day_type,
                         'RecentLag': same_day_hour_moving_average,
                         'PreviousYearLoadLag':  same_week_day_hour_lag,
                         'PreviousYearTempLag':  same_day_hour_lag,
                         'LoadRatio': compute_load_ratio}


def parse_feature_config(feature_config, feature_function_dict):
    """
    A helper function parsing a feature_config to feature name, column to
    compute the feature on, feature function to use, and arguments to the
    feature function
    """
    feature_name = feature_config[0]
    feature_col = feature_config[1]
    feature_args = feature_config[2]
    feature_function = feature_function_dict[feature_name]

    return feature_name, feature_col, feature_args, feature_function


def create_basic_features(input_df, datetime_colname, basic_feature_list):
    """
    This helper function uses the functions in common.feature_utils to
    create a set of basic features which are independently created for each
    row, i.e. no lag features or rolling window features.

    Args:
        input_df (pandas.DataFrame): data frame for which to compute basic features.
        datetime_colname (str): name of Datetime column

    Returns:
        pandas.DataFrame: output data frame which contains newly created features
    """
    output_df = input_df.copy()
    if not is_datetime_like(output_df[datetime_colname]):
        output_df[datetime_colname] = \
            pd.to_datetime(output_df[datetime_colname], format=DATETIME_FORMAT)
    datetime_col = output_df[datetime_colname]

    for feature_config in basic_feature_list:
        feature_name, feature_col, feature_args, feature_function = \
            parse_feature_config(feature_config, feature_function_dict)
        feature = feature_function(datetime_col, **feature_args)

        if isinstance(feature, np.ndarray) or isinstance(feature, pd.Series):
            output_df[feature_name] = feature
        else:
            for k, v in feature.items():
                output_df[k] = v

    return output_df


def create_advanced_features(train_df, test_df,
                             advanced_feature_list,
                             lag_feature_list):
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
        holiday_colname (str): name of Holiday column (if present), default value is None

    Returns:
        pandas.DataFrame: output containing newly constructed features on training data
        pandas.DataFrame: output containing newly constructed features on testing data

    """
    output_df = pd.concat([train_df, test_df], sort=True)
    if not is_datetime_like(output_df[DATETIME_COLNAME]):
        output_df[DATETIME_COLNAME] = \
            pd.to_datetime(output_df[DATETIME_COLNAME], format=DATETIME_FORMAT)
    datetime_col = output_df[DATETIME_COLNAME]
    forecast_creation_time = max(train_df[DATETIME_COLNAME])

    for feature_config in advanced_feature_list:
        feature_name, feature_col, feature_args, feature_function = \
            parse_feature_config(feature_config, feature_function_dict)
        if feature_col is None:
            feature = feature_function(datetime_col, **feature_args)
        else:
            feature = feature_function(datetime_col, output_df[feature_col])
        output_df[feature_name] = feature

    load_ratio_flag = False
    feature_df_list = []
    for feature_config in lag_feature_list:
        feature_name, feature_col, feature_args, feature_function = \
            parse_feature_config(feature_config, feature_function_dict)
        if feature_name != 'LoadRatio':
            if 'forecast_creation_time' in feature_args.keys():
                feature_args['forecast_creation_time'] = forecast_creation_time

            feature = output_df[[DATETIME_COLNAME, feature_col, GRAIN_COLNAME]]\
                .groupby(GRAIN_COLNAME)\
                .apply(lambda g: feature_function(g[DATETIME_COLNAME],
                                                  g[feature_col],
                                                  **feature_args)
                       )
            feature.reset_index(inplace=True)
            feature_df_list.append(feature)
        else:
            load_ratio_flag = True
            feature_name, feature_col, feature_args, feature_function = \
                parse_feature_config(feature_config, feature_function_dict)
            load_ratio_args = feature_args

    output_df = reduce(
        lambda left, right: pd.merge(left, right,
                                     on=[DATETIME_COLNAME, GRAIN_COLNAME]),
        [output_df] + feature_df_list)

    if load_ratio_flag:
        output_df = compute_load_ratio(output_df, **load_ratio_args)

    # Split train and test data and return separately
    train_end = max(train_df[DATETIME_COLNAME])
    output_df_train = output_df.loc[output_df[DATETIME_COLNAME] <= train_end,]
    output_df_test = output_df.loc[output_df[DATETIME_COLNAME] > train_end,]

    return output_df_train, output_df_test


def compute_features(train_dir, test_dir, output_dir, basic_feature_list,
                     advanced_feature_list, lag_feature_list,
                     filter_by_month=True):
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
                                parse_dates=[DATETIME_COLNAME])

    # These features only need to be created once for all rounds
    train_base_basic_features = \
        create_basic_features(train_base_df,
                              datetime_colname=DATETIME_COLNAME,
                              basic_feature_list=basic_feature_list)

    for i in range(1, NUM_ROUND + 1):
        train_file = os.path.join(train_dir,
                                  TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_file = os.path.join(test_dir, TEST_FILE_PREFIX + str(i) + '.csv')

        train_delta_df = pd.read_csv(train_file, parse_dates=[DATETIME_COLNAME])
        test_round_df = pd.read_csv(test_file, parse_dates=[DATETIME_COLNAME])

        train_delta_basic_features = \
            create_basic_features(train_delta_df,
                                  datetime_colname=DATETIME_COLNAME,
                                  basic_feature_list=basic_feature_list)
        test_basic_features = \
            create_basic_features(test_round_df,
                                  datetime_colname=DATETIME_COLNAME,
                                  basic_feature_list=basic_feature_list)

        train_round_df = pd.concat([train_base_df, train_delta_df])
        train_advanced_features, test_advanced_features = \
            create_advanced_features(train_round_df, test_round_df,
                                     advanced_feature_list=advanced_feature_list,
                                     lag_feature_list=lag_feature_list)

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

        train_all_features.dropna(inplace=True)
        test_all_features.drop(['DewPnt', 'DryBulb', 'DEMAND'],
                               inplace=True, axis=1)

        if filter_by_month:
            test_month = test_basic_features['MonthOfYear'].values[0]
            train_all_features = train_all_features.loc[
                train_all_features['MonthOfYear'] == test_month, ].copy()

        train_output_file = os.path.join(output_dir, 'train',
                                         TRAIN_FILE_PREFIX + str(i) + '.csv')
        test_output_file = os.path.join(output_dir, 'test',
                                        TEST_FILE_PREFIX + str(i) + '.csv')

        train_all_features.to_csv(train_output_file, index=False)
        test_all_features.to_csv(test_output_file, index=False)

        print('Round {}'.format(i))
        print('Training data size: {}'.format(train_all_features.shape))
        print('Testing data size: {}'.format(test_all_features.shape))
        print('Minimum training timestamp: {}'.
              format(min(train_all_features[DATETIME_COLNAME])))
        print('Maximum training timestamp: {}'.
              format(max(train_all_features[DATETIME_COLNAME])))
        print('Minimum testing timestamp: {}'.
              format(min(test_all_features[DATETIME_COLNAME])))
        print('Maximum testing timestamp: {}'.
              format(max(test_all_features[DATETIME_COLNAME])))
        print('')