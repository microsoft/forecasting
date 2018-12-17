# import packages
import os
import inspect
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.training as training
import argparse

from rnn_train import rnn_train
from rnn_predict import rnn_predict
from make_features import make_features
import hparams
from utils import *
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

data_relative_dir = '../../data'


def create_round_prediction(data_dir, submission_round, hparams, make_features_flag=True, train_model_flag=True, train_back_offset=0,
                            predict_cut_mode='predict', random_seed=1):
    # conduct feature engineering and save related numpy array to disk
    if make_features_flag:
        make_features(submission_round=submission_round)

    # read the numpy arrays output from the make_features.py
    # file_dir = './prototypes/retail_rnn_model'

    intermediate_data_dir = os.path.join(data_dir, 'intermediate/round_{}'.format(submission_round))

    ts_value_train = np.load(os.path.join(intermediate_data_dir, 'ts_value_train.npy'))
    feature_train = np.load(os.path.join(intermediate_data_dir, 'feature_train.npy'))
    feature_test = np.load(os.path.join(intermediate_data_dir, 'feature_test.npy'))

    # convert the dtype to float32 to suffice tensorflow cudnn_rnn requirements.
    ts_value_train = ts_value_train.astype(dtype='float32')
    feature_train = feature_train.astype(dtype='float32')
    feature_test = feature_test.astype(dtype='float32')

    # define parameters
    # constant
    predict_window = feature_test.shape[1]

    # train the rnn model
    if train_model_flag:
        tf.reset_default_graph()
        tf.set_random_seed(seed=random_seed)
        train_error = rnn_train(ts_value_train, feature_train, feature_test, hparams, predict_window,
                                intermediate_data_dir, submission_round, back_offset=train_back_offset)


    # make prediction
    tf.reset_default_graph()
    pred_batch_size = 1024
    pred_o = rnn_predict(ts_value_train, feature_train, feature_test, hparams, predict_window, intermediate_data_dir,
                         submission_round, pred_batch_size, cut_mode=predict_cut_mode)
    return pred_o, train_error


def create_round_submission(data_dir, submission_round, hparams, make_features_flag=True, train_model_flag=True, train_back_offset=0,
                            predict_cut_mode='predict', random_seed=1):

    pred_o, _ = create_round_prediction(data_dir, submission_round, hparams, make_features_flag=make_features_flag,
                                        train_model_flag=train_model_flag, train_back_offset=train_back_offset,
                                        predict_cut_mode=predict_cut_mode, random_seed=random_seed)
    # get rid of prediction at horizon 1
    pred_sub = pred_o[:, 1:].reshape((-1))

    # arrange the predictions into pd.DataFrame
    # read in the test_file for this round
    train_file = os.path.join(data_dir, 'train/train_round_{}.csv'.format(submission_round))
    test_file = os.path.join(data_dir, 'train/aux_round_{}.csv'.format(submission_round))

    train = pd.read_csv(train_file, index_col=False)
    test = pd.read_csv(test_file, index_col=False)

    train_last_week = bs.TRAIN_END_WEEK_LIST[submission_round- 1]

    store_list = train['store'].unique()
    brand_list = train['brand'].unique()
    test_week_list = range(bs.TEST_START_WEEK_LIST[submission_round - 1], bs.TEST_END_WEEK_LIST[submission_round - 1] + 1)

    test_item_list = list(itertools.product(store_list, brand_list, test_week_list))
    test_item_df = pd.DataFrame.from_records(test_item_list, columns=['store', 'brand', 'week'])

    test = test_item_df.merge(test, how='left', on=['store', 'brand', 'week'])

    submission = test.sort_values(by=['store', 'brand', 'week'], ascending=True)
    submission['round'] = submission_round
    submission['weeks_ahead'] = submission['week'] - train_last_week
    submission['prediction'] = pred_sub
    submission = submission[['round', 'store', 'brand', 'week', 'weeks_ahead', 'prediction']]

    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, dest='seed', default=1, help='random seed')
    args = parser.parse_args()
    random_seed = args.seed


    # set the data directory
    file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    data_dir = os.path.join(file_dir, data_relative_dir)

    # import hyper parameters
    # TODO: add ema in the code to imporve the performance
    hparams_dict = hparams.hparams_smac
    hparams = training.HParams(**hparams_dict)
    num_round = len(bs.TEST_END_WEEK_LIST)
    pred_all = pd.DataFrame()
    for R in range(1, num_round + 1):
        print('create submission for round {}...'.format(R))
        round_submission = create_round_submission(data_dir, R, hparams, random_seed=random_seed)
        pred_all = pred_all.append(round_submission)

    file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    submission_relative_dir = '../'
    submission_dir = os.path.join(file_dir, submission_relative_dir)
    if not os.path.isdir(submission_dir):
        os.makedirs(submission_dir)

    submission_file = os.path.join(submission_dir, 'submission_seed_{}.csv'.format(str(random_seed)))
    pred_all.to_csv(submission_file, index=False)


