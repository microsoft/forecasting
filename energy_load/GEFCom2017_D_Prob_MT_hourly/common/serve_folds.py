import os, sys, inspect
import pandas as pd

def serve_folds():
    """
    This script generates train and test data for each forecast round on the fly
    """
    
    from benchmark_paths import DATA_DIR
    TRAIN_DATA_DIR = DATA_DIR + '/train'
    TEST_DATA_DIR = DATA_DIR + '/test'
    # This file stores all the data before 2016-12-01
    TRAIN_BASE_FILE = 'train_base.csv'
    # These files contain data to be added to train_base.csv to form the training data of a particular round
    TRAIN_ROUND_FILE_PREFIX = 'train_round_'
    TEST_ROUND_FILE_PREFIX = 'test_round_'

    NUM_FOLDS = 6

    train_base = pd.read_csv(os.path.join(TRAIN_DATA_DIR, TRAIN_BASE_FILE))

    for i in range(NUM_FOLDS):
        train_round_file_name = TRAIN_ROUND_FILE_PREFIX + str(i+1) + '.csv'
        train_round_delta = \
            pd.read_csv(os.path.join(TRAIN_DATA_DIR, train_round_file_name))

        train_round_df = pd.concat([train_base, train_round_delta])

        test_round_file_name = TEST_ROUND_FILE_PREFIX + str(i+1) + '.csv'

        test_round_df = \
            pd.read_csv(os.path.join( TEST_DATA_DIR, test_round_file_name))

        yield train_round_df, test_round_df, i+1

# Test serve_folds
for train, test, _ in serve_folds():
    print('Training data size: {}'.format(train.shape))
    print('Testing data size: {}'.format(test.shape))
    print('Minimum training timestamp: {}'.format(min(train['Datetime'])))
    print('Maximum training timestamp: {}'.format(max(train['Datetime'])))
    print('Minimum testing timestamp: {}'.format(min(test['Datetime'])))
    print('Maximum testing timestamp: {}'.format(max(test['Datetime'])))
    print('')
