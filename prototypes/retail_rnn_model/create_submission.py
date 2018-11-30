# import packages
import os
import inspect
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.training as training

from rnn_train import rnn_train
from rnn_predict import rnn_predict
from make_features import make_features
import hparams
from utils import *
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs


def create_round_submission(ROUND, hparams):
    # import hyper parameters
    # TODO: add ema in the code to imporve the performance

    # conduct feature engineering and save related numpy array to disk
    make_features(round=ROUND)

    # read the numpy arrays output from the make_features.py
    # file_dir = './prototypes/retail_rnn_model'
    file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    data_relative_dir = '../../retail_sales/OrangeJuice_Pt_3Weeks_Weekly/data'
    data_dir = os.path.join(file_dir, data_relative_dir)
    intermediate_data_dir = os.path.join(data_dir, 'intermediate/round_{}'.format(ROUND))

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
    tf.reset_default_graph()
    rnn_train(ts_value_train, feature_train, feature_test, hparams, predict_window, intermediate_data_dir, ROUND)

    # make prediction
    tf.reset_default_graph()
    pred_batch_size = 1024
    pred_o = rnn_predict(ts_value_train, feature_train, feature_test, hparams, predict_window, intermediate_data_dir,
                         ROUND, pred_batch_size)

    # get rid of prediction at horizon 1
    pred_sub = pred_o[:, 1:].reshape((-1))

    # arrange the predictions into pd.DataFrame
    # read in the test_file for this round
    test_file = os.path.join(data_dir, 'train/aux_round_{}.csv'.format(ROUND))
    test = pd.read_csv(test_file, index_col=False)
    train_last_week = bs.TRAIN_END_WEEK_LIST[ROUND - 1]
    test = test.loc[test['week'] > train_last_week + 1]
    test = test.groupby(['store', 'brand']).apply(
        lambda df: fill_datetime_gap(df, min_time=bs.TEST_START_WEEK_LIST[ROUND - 1],
                                     max_time=bs.TEST_END_WEEK_LIST[ROUND - 1]))
    test = test.reset_index(level=[0, 1]).reset_index(drop=True)

    submission = test.sort_values(by=['store', 'brand', 'week'], ascending=True)
    submission['round'] = ROUND
    submission['weeks_ahead'] = submission['week'] - train_last_week
    submission['prediction'] = pred_sub
    submission = submission[['round', 'store', 'brand', 'week', 'weeks_ahead', 'prediction']]

    return submission


if __name__ == '__main__':
    hparams_dict = hparams.hparams_manual
    hparams = training.HParams(**hparams_dict)
    num_round = len(bs.TEST_END_WEEK_LIST)
    pred_all = pd.DataFrame()
    for R in range(1, num_round + 1):
        print('create submission for round {}...'.format(R))
        round_submission = create_round_submission(R, hparams)
        pred_all.append(round_submission)
    print(1)


