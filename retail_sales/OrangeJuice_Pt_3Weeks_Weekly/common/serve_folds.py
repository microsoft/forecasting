# Return training and testing data for each forecast round
# 
# You can use this script in either of the following two ways
# 1. Import the serve_folds module from this script to generate the training and testing data for 
# each forecast period on the fly
# 2. Run the script using the syntax below
#    python serve_folds [-h] [--test] [--save]
# where if '--test' is specified a quick test of serve_folds module will run and furthermore if 
# `--save' is specified the training and testing data will be saved as csv files. Note that '--save' 
# is effective only if '--test' is specified. This means that you need to run
#    python serve_folds --test --save 
# to get the output data files stored in /train and /test folders under the data directory. 

import os
import sys
import inspect
import argparse
import pandas as pd
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

def serve_folds(write_csv=False): 
    # Get the directory of this script and directory of the OrangeJuice dataset
    SCRIPT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'data')
    # Read sales data into dataframe
    sales = pd.read_csv(os.path.join(DATA_DIR, 'yx.csv'), index_col=0)

    if write_csv:
        TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
        TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
        if not os.path.isdir(TRAIN_DATA_DIR):
            os.mkdir(TRAIN_DATA_DIR)
        if not os.path.isdir(TEST_DATA_DIR):
            os.mkdir(TEST_DATA_DIR)

    for i in range(bs.NUM_ROUNDS):
        data_mask = (sales.week>=bs.TRAIN_START_WEEK) & (sales.week<=bs.TRAIN_END_WEEK_LIST[i])
        train = sales[data_mask].copy()
        data_mask = (sales.week>=bs.TEST_START_WEEK_LIST[i]) & (sales.week<=bs.TEST_END_WEEK_LIST[i])
        test = sales[data_mask].copy()
        if write_csv:
            train.to_csv(os.path.join(TRAIN_DATA_DIR, 'train_round_' + str(i+1) + '.csv'))
            test.to_csv(os.path.join(TEST_DATA_DIR, 'test_round_' + str(i+1) + '.csv'))
        yield train, test

# Test serve_folds
parser = argparse.ArgumentParser()
parser.add_argument('--test', help='Run the test of serve_folds function', action='store_true')
parser.add_argument('--save', help='Write training and testing data into csv files', action='store_true')
args = parser.parse_args()
if args.test:
    for train, test in serve_folds(args.save):    
        print('Training data size: {}'.format(train.shape))
        print('Testing data size: {}'.format(test.shape))
        print('Minimum training week number: {}'.format(min(train['week'])))
        print('Maximum training week number: {}'.format(max(train['week'])))
        print('Minimum testing week number: {}'.format(min(test['week'])))
        print('Maximum testing week number: {}'.format(max(test['week'])))
        print('')
