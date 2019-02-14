"""
This script creates a set of commonly used features using the functions in
common.feature_utils, which serve as a set of baseline features.
Feel free to write your own feature engineering code to create new features by
calling the feature_utils functions with alternative parameters.
"""
import os, sys, getopt

import localpath
from energy_load.GEFCom2017_D_Prob_MT_hourly.common.benchmark_paths \
    import DATA_DIR, SUBMISSIONS_DIR
from energy_load.GEFCom2017_D_Prob_MT_hourly.common.benchmark_settings import\
    DATETIME_COLNAME, GRAIN_COLNAME, HOLIDAY_COLNAME
from energy_load.GEFCom2017_D_Prob_MT_hourly.common.compute_features\
    import compute_features

print('Data directory used: {}'.format(DATA_DIR))

OUTPUT_DIR = os.path.join(DATA_DIR, 'features')
TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_BASE_FILE = 'train_base.csv'
TRAIN_FILE_PREFIX = 'train_round_'
TEST_FILE_PREFIX = 'test_round_'
NUM_ROUND = 6


DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Feature lists used to specify the features to be computed by compute_features.
# The reason we have three lists is that they are handled differently by
# compute_features.
# Each feature list includes a list of "feature configurations".
# Each feature configuration is tuple in the format of (FeatureName,
# FeatureCol, FeatureArgs)
# FeatureName is used to determine the function to use,
# see feature_function_dict in compute_features
# FeatureCol is a string specifying the column to compute a feature on. It
# can be done for features that only requires the datetime column
# FeatureArgs is a dictionary of additional arguments passed to the feature
# function
basic_feature_list = [('Hour', None, {}),
                      ('DayOfWeek', None, {}),
                      ('DayOfMonth', None, {}),
                      ('TimeOfYear', None, {}),
                      ('WeekOfYear', None, {}),
                      ('MonthOfYear', None, {}),
                      ('AnnualFourier', None, {'n_harmonics': 3}),
                      ('WeeklyFourier', None, {'n_harmonics': 3}),
                      ('DailyFourier', None, {'n_harmonics': 2})]


advanced_feature_list = [('CurrentDate', None, {}),
                         ('CurrentDateHour', None, {}),
                         ('CurrentYear', None, {}),
                         ('DayType', HOLIDAY_COLNAME, {})]

lag_feature_list = [('PreviousYearLoadLag', 'DEMAND',
                     {'output_colname': 'LoadLag'}),
                    ('PreviousYearTempLag', 'DewPnt',
                     {'output_colname': 'DewPntLag'}),
                    ('PreviousYearTempLag', 'DryBulb',
                     {'output_colname': 'DryBulbLag'}),
                    ('RecentLag', 'DEMAND',
                     {'start_week': 10,
                      'window_size': 4,
                      'average_count': 8,
                      'forecast_creation_time': None,
                      'output_col_prefix': 'RecentLoad_'}),
                    ('RecentLag', 'DryBulb',
                     {'start_week': 9,
                      'window_size': 4,
                      'average_count': 8,
                      'forecast_creation_time': None,
                      'output_col_prefix': 'RecentDryBulb_'}),
                    ('RecentLag', 'DewPnt',
                     {'start_week': 9,
                      'window_size': 4,
                      'average_count': 8,
                      'forecast_creation_time': None,
                      'output_col_prefix': 'RecentDewPnt_'})]


def main(train_dir, test_dir, output_dir):
    compute_features(train_dir, test_dir, output_dir, basic_feature_list,
                     advanced_feature_list, lag_feature_list)


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

    compute_features(TRAIN_DATA_DIR, TEST_DATA_DIR, OUTPUT_DIR,
                     basic_feature_list, advanced_feature_list,
                     lag_feature_list,filter_by_month=False)
