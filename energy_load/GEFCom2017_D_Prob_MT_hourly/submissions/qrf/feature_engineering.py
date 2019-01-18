"""
This script creates a set of commonly used features using the functions in
common.feature_utils, which serve as a set of baseline features.
Feel free to write your own feature engineering code to create new features by
calling the feature_utils functions with alternative parameters.
"""
import os, sys, getopt
from functools import reduce

import localpath
from energy_load.GEFCom2017_D_Prob_MT_hourly.common.benchmark_paths import DATA_DIR, SUBMISSIONS_DIR
from common.feature_utils import *
from common.utils import is_datetime_like

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

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def create_basic_features(input_df, datetime_colname):
    """
    This helper function uses the functions in common.feature_utils to
    create a set of basic features which are independently created for each
    row, i.e. no lag features or rolling window features.
    """

    output_df = input_df.copy()
    if not is_datetime_like(output_df[datetime_colname]):
        output_df[datetime_colname] = \
            pd.to_datetime(output_df[datetime_colname], format=DATETIME_FORMAT)
    datetime_col = output_df[datetime_colname]

    # Basic temporal features
    output_df['Hour'] = hour_of_day(datetime_col)
    output_df['DayOfWeek'] = day_of_week(datetime_col)   
    output_df['DayOfMonth'] = day_of_month(datetime_col) 
    output_df['TimeOfYear'] = time_of_year(datetime_col)
    output_df['WeekOfYear'] = week_of_year(datetime_col)
    output_df['MonthOfYear'] = month_of_year(datetime_col)
    
    # Fourier approximation features
    annual_fourier_approx = annual_fourier(datetime_col, n_harmonics=3)
    weekly_fourier_approx = weekly_fourier(datetime_col, n_harmonics=3)
    daily_fourier_approx = daily_fourier(datetime_col, n_harmonics=2)

    for k, v in annual_fourier_approx.items():
        output_df[k] = v

    for k, v in weekly_fourier_approx.items():
        output_df[k] = v

    for k, v in daily_fourier_approx.items():
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
    """
    output_df = pd.concat([train_df, test_df], sort=True)
    if not is_datetime_like(output_df[datetime_colname]):
        output_df[datetime_colname] = \
            pd.to_datetime(output_df[datetime_colname], format=DATETIME_FORMAT)
    datetime_col = output_df[datetime_colname]
    forecast_creation_time = max(train_df[datetime_colname])

    # Temporal features indicating the position of a record in the entire
    # time period under consideration.
    # For example, if the first date in training data is 2011-01-01 and the
    # last date in testing data is 2017-02-28. The 'CurrentDate' feature
    # for 2011-01-01 is 0, and for 2017-02-28 is 1.
    min_date = min(datetime_col.dt.date)
    max_date = max(datetime_col.dt.date)
    output_df['CurrentDate'] = \
        normalized_current_date(datetime_col, min_date, max_date)

    # 'CurrentDateHour' is similar to 'CurrentDate', and at hour level.
    min_datehour = min(datetime_col)
    max_datehour = max(datetime_col)
    output_df['CurrentDateHour'] = \
        normalized_current_datehour(datetime_col, min_datehour, max_datehour)

    # 'CurrentYear' is similar to 'CurrentDate', and at year level.
    min_year = min(datetime_col.dt.year)
    max_year = max(datetime_col.dt.year)
    output_df['CurrentYear'] = normalized_current_year(
        datetime_col, min_year, max_year)

    output_df['DayType'] = day_type(datetime_col, output_df[holiday_colname])

    # Load lag feature based on previous years' load
    same_week_day_hour_load_lag = \
        output_df[[datetime_colname, 'DEMAND', 'Zone']].groupby('Zone').apply(
            lambda g: same_week_day_hour_lag(g[datetime_colname],
                                             g['DEMAND'],
                                             output_colname='LoadLag'))
    same_week_day_hour_load_lag.reset_index(inplace=True)

    # Temperature lag features based on previous years' temperature
    same_day_hour_drewpnt_lag = \
        output_df[[datetime_colname, 'DewPnt', 'Zone']].groupby('Zone').apply(
            lambda g: same_day_hour_lag(g[datetime_colname], g['DewPnt'],
                                        output_colname='DewPntLag'))
    same_day_hour_drewpnt_lag.reset_index(inplace=True)

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
                                                   average_count=8,
                                                   forecast_creation_time=forecast_creation_time,
                                                   output_col_prefix='RecentLoad_'))
    load_moving_average.reset_index(inplace=True)

    # Moving average features of dew point of recent weeks
    dewpnt_moving_average = \
        output_df[[datetime_colname, 'DewPnt', 'Zone']].groupby('Zone').apply(
            lambda g: same_day_hour_moving_average(g[datetime_colname],
                                                   g['DewPnt'],
                                                   start_week=9,
                                                   window_size=4,
                                                   average_count=8,
                                                   forecast_creation_time=forecast_creation_time,
                                                   output_col_prefix='RecentDewPnt_'))
    dewpnt_moving_average.reset_index(inplace=True)

    # Moving average features of dry bulb of recent weeks
    drybulb_moving_average = \
        output_df[[datetime_colname, 'DryBulb', 'Zone']].groupby('Zone').apply(
            lambda g: same_day_hour_moving_average(g[datetime_colname],
                                                   g['DryBulb'],
                                                   start_week=9,
                                                   window_size=4,
                                                   average_count=8,
                                                   forecast_creation_time=forecast_creation_time,
                                                   output_col_prefix='RecentDryBulb_'))
    drybulb_moving_average.reset_index(inplace=True)

    # Put everything together
    output_df = reduce(
        lambda left, right: pd.merge(left, right, on=[datetime_colname, 'Zone']),
        [output_df, same_week_day_hour_load_lag,
         same_day_hour_drewpnt_lag, same_day_hour_drybulb_lag,
         load_moving_average, drybulb_moving_average, dewpnt_moving_average])

    # Split train and test data and return separately
    train_end = max(train_df[datetime_colname])
    output_df_train = output_df.loc[output_df[datetime_colname] <= train_end, ]
    output_df_test = output_df.loc[output_df[datetime_colname] > train_end, ]

    return output_df_train, output_df_test


def main(train_dir, test_dir, output_dir, datetime_colname, holiday_colname):
    """
    This helper function uses the create_basic_features and create_advanced
    features functions to create features for each train and test round.
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

        train_all_features.dropna(inplace=True)
        test_all_features.drop(['DewPnt', 'DryBulb', 'DEMAND'],
                               inplace=True, axis=1)

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
            output_data_dir = os.path.join(SUBMISSIONS_DIR, submission_folder,
                                           'data')
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
