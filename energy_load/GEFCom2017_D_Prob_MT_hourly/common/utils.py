import os

from benchmark_settings import TRAIN_BASE_END, TRAIN_ROUNDS_ENDS, \
    TEST_STARTS_ENDS

# These columns should be dropped in the test data as they are not available
# at forecasting time
DROP_COLUMNS = ['DEMAND', 'DewPnt', 'DryBulb']


def split_train_test(full_df, output_dir,
                     train_base_file='train_base.csv',
                     train_file_prefix='train_round_',
                     test_file_prefix='test_round_'):
    """
    This helper function splits full_df into train and test folds as defined
    in benchmark_settings.py
    :param full_df:
        Full data frame to be split. For robustness and
        simplicity, it's required that full_df is indexed by datetime at
        level 0.
    :param output_dir: Directory to store the output files in.
    :param datetime_colname: Column or index in full_df used for splitting.
    :param train_base_file:
        This file stores all the data before TRAIN_BASE_END.
    :param train_file_prefix:
        These files each contains a subset of data to be added to
        train_base_file to form the training data of a particular round.
    :param test_file_prefix:
        These files each contains testing data for a particular round
    """

    train_data_dir = os.path.join(output_dir, 'train')
    test_data_dir = os.path.join(output_dir, 'test')
    ground_truth_dir = os.path.join(output_dir, 'test_ground_truth')

    # Create train and test data directories
    if not os.path.isdir(train_data_dir):
        os.mkdir(train_data_dir)

    if not os.path.isdir(test_data_dir):
        os.mkdir(test_data_dir)

    if not os.path.isdir(ground_truth_dir):
        os.mkdir(ground_truth_dir)

    index_value = full_df.index.get_level_values(0)
    train_base_df = full_df.loc[index_value < TRAIN_BASE_END].copy()

    train_base_df.to_csv(os.path.join(train_data_dir, train_base_file))
    print('Base training data frame size: {}'.format(train_base_df.shape))

    for i in range(len(TRAIN_ROUNDS_ENDS)):
        file_name = os.path.join(train_data_dir,
                                 train_file_prefix + str(i+1) + '.csv')
        train_round_delta_df = full_df.loc[
            (index_value >= TRAIN_BASE_END)
            & (index_value < TRAIN_ROUNDS_ENDS[i])]
        print('Round {0} additional training data size: {1}'
              .format(i+1, train_round_delta_df.shape))
        print('Minimum timestamp: {0}'
              .format(min(train_round_delta_df.index.get_level_values(0))))
        print('Maximum timestamp: {0}'
              .format(max(train_round_delta_df.index.get_level_values(0))))
        print('')
        train_round_delta_df.to_csv(file_name)

    for i in range(len(TEST_STARTS_ENDS)):
        test_file = os.path.join(test_data_dir,
                                 test_file_prefix + str(i+1) + '.csv')
        ground_truth_file = os.path.join(ground_truth_dir,
                                         test_file_prefix + str(i+1) + '.csv')
        start_end = TEST_STARTS_ENDS[i]
        test_round_df = full_df.loc[
            ((index_value >= start_end[0]) & (index_value < start_end[1]))
        ].copy()

        test_round_df.to_csv(ground_truth_file)
        test_round_df.drop(DROP_COLUMNS, inplace=True, axis=1)
        test_round_df.to_csv(test_file)

        print('Round {0} testing data size: {1}'
              .format(i+1, test_round_df.shape))
        print('Minimum timestamp: {0}'.format(min(
            test_round_df.index.get_level_values(0))))
        print('Maximum timestamp: {0}'.format(max(
            test_round_df.index.get_level_values(0))))
        print('')

