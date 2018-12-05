# import packages
import os
import inspect
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.training as training

from create_submission import create_round_prediction
import hparams
from utils import *
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs
from common.metrics import MAPE


def eval_function(hparams_dict, data_dir):
    hparams = training.HParams(**hparams_dict)
    # use round 1 training data for hyper parameter tuning to avoid data leakage for later rounds
    submission_round = 1
    make_features_flag = False  # no need to make feature since it is alreayd made and saved to disk
    train_model_flag = True
    train_back_offset = 3  # equal to predict_window
    predict_cut_mode = 'eval'

    # get prediction
    pred_o, train_mape = create_round_prediction(data_dir, submission_round, hparams, make_features_flag=make_features_flag,
                                     train_model_flag=train_model_flag, train_back_offset=train_back_offset,
                                     predict_cut_mode=predict_cut_mode)
    # get rid of prediction at horizon 1
    pred_sub = pred_o[:, 1:].reshape((-1))

    # evaluate the prediction on last two days in the first round training data
    # TODO: get train error and evalution error for different parameters
    train_file = os.path.join(data_dir, 'train/train_round_{}.csv'.format(submission_round))
    train = pd.read_csv(train_file, index_col=False)
    train_last_week = bs.TRAIN_END_WEEK_LIST[submission_round - 1]
    # filter the train to contain ony last two days' data
    train = train.loc[train['week'] >= train_last_week - 1]

    # create the data frame without missing dates
    store_list = train['store'].unique()
    brand_list = train['brand'].unique()
    week_list = range(train_last_week - 1, train_last_week + 1)
    item_list = list(itertools.product(store_list, brand_list, week_list))
    item_df = pd.DataFrame.from_records(item_list, columns=['store', 'brand', 'week'])

    train = item_df.merge(train, how='left', on=['store', 'brand', 'week'])
    result = train.sort_values(by=['store', 'brand', 'week'], ascending=True)
    result['prediction'] = pred_sub
    result['sales'] = result['logmove'].apply(lambda x: round(np.exp(x)))

    # calculate MAPE on the evaluate set
    result = result.loc[result['sales'].notnull()]
    eval_mape = MAPE(result['prediction'], result['sales'])
    return train_mape, eval_mape



if __name__ == '__main__':
    # set the data directory
    file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    data_relative_dir = '../../retail_sales/OrangeJuice_Pt_3Weeks_Weekly/data'
    data_dir = os.path.join(file_dir, data_relative_dir)

    # import hyper parameters
    # TODO: add ema in the code to imporve the performance
    hparams_dict = hparams.hparams_manual
    eval_function(hparams_dict, data_dir)

    print(1)






