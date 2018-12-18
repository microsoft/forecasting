# import packages
import os
import inspect
import itertools
import sys
import numpy as np
import pandas as pd
import tensorflow.contrib.training as training

from train_score import create_round_prediction
from utils import *

# Add TSPerf root directory to sys.path
file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
tsperf_dir = os.path.join(file_dir, '../../../../')

if tsperf_dir not in sys.path:
    sys.path.append(tsperf_dir)

import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs
from common.metrics import MAPE

from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.facade.smac_facade import SMAC


LIST_HYPERPARAMETER = ['decoder_input_dropout', 'decoder_state_dropout', 'decoder_output_dropout']
data_relative_dir = '../../data'


def eval_function(hparams_dict):
    # set the data directory
    file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    data_dir = os.path.join(file_dir, data_relative_dir)

    hparams_dict = dict(hparams_dict)
    for key in LIST_HYPERPARAMETER:
        hparams_dict[key] = [hparams_dict[key]]

    # add the value of other hyper parameters which are not tuned
    hparams_dict['encoder_rnn_layers'] = 1
    hparams_dict['decoder_rnn_layers'] = 1
    hparams_dict['decoder_variational_dropout'] = [False]
    hparams_dict['asgd_decay'] = None

    hparams = training.HParams(**hparams_dict)
    # use round 1 training data for hyper parameter tuning to avoid data leakage for later rounds
    submission_round = 1
    make_features_flag = False
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
    return eval_mape


if __name__ == '__main__':

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()

    # add parameters to the configuration space
    train_window = UniformIntegerHyperparameter('train_window', 3, 70, default_value=60)
    cs.add_hyperparameter(train_window)

    batch_size = CategoricalHyperparameter('batch_size', [64, 128, 256, 1024], default_value=64)
    cs.add_hyperparameter(batch_size)

    rnn_depth = UniformIntegerHyperparameter('rnn_depth', 100, 500, default_value=400)
    cs.add_hyperparameter(rnn_depth)

    encoder_dropout = UniformFloatHyperparameter('encoder_dropout', 0.0, 0.05, default_value=0.03)
    cs.add_hyperparameter(encoder_dropout)

    gate_dropout = UniformFloatHyperparameter('gate_dropout', 0.95, 1.0, default_value=0.997)
    cs.add_hyperparameter(gate_dropout)

    decoder_input_dropout = UniformFloatHyperparameter('decoder_input_dropout', 0.95, 1.0, default_value=1.0)
    cs.add_hyperparameter(decoder_input_dropout)

    decoder_state_dropout = UniformFloatHyperparameter('decoder_state_dropout', 0.95, 1.0, default_value=0.99)
    cs.add_hyperparameter(decoder_state_dropout)

    decoder_output_dropout = UniformFloatHyperparameter('decoder_output_dropout', 0.95, 1.0, default_value=0.975)
    cs.add_hyperparameter(decoder_output_dropout)

    max_epoch = CategoricalHyperparameter('max_epoch', [20, 50, 100], default_value=20)
    cs.add_hyperparameter(max_epoch)

    learning_rate = CategoricalHyperparameter('learning_rate', [0.001, 0.01, 0.1], default_value=0.001)
    cs.add_hyperparameter(learning_rate)

    beta1 = UniformFloatHyperparameter('beta1', 0.5, 0.9999, default_value=0.9)
    cs.add_hyperparameter(beta1)

    beta2 = UniformFloatHyperparameter('beta2', 0.5, 0.9999, default_value=0.999)
    cs.add_hyperparameter(beta2)

    epsilon = CategoricalHyperparameter('epsilon', [1e-08, 0.00001, 0.0001, 0.1, 1], default_value=1e-08)
    cs.add_hyperparameter(epsilon)

    scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                         "runcount-limit": 50,  # maximum function evaluations
                         "cs": cs,  # configuration space
                         "deterministic": "true"
                         })



    # import hyper parameters
    # TODO: add ema in the code to imporve the performance
    smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=eval_function)

    incumbent = smac.optimize()
    inc_value = eval_function(incumbent)
    print('the best hyper parameter sets are:')
    print(incumbent)
    print('the corresponding MAPE is: {}'.format(inc_value))

    # following are the print out:
    # the best hyper parameter sets are:
    # Configuration:
    # batch_size, Value: 64
    # beta1, Value: 0.7763754022206656
    # beta2, Value: 0.7923825287287111
    # decoder_input_dropout, Value: 0.9975650671957902
    # decoder_output_dropout, Value: 0.9732177111192211
    # decoder_state_dropout, Value: 0.9743711264734845
    # encoder_dropout, Value: 0.024688459483309007
    # epsilon, Value: 1e-08
    # gate_dropout, Value: 0.980832247298109
    # learning_rate, Value: 0.001
    # max_epoch, Value: 100
    # rnn_depth, Value: 387
    # train_window, Value: 26

    # the corresponding MAPE is: 0.36703585613035433



