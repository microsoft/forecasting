# import packages
import os
import inspect
import numpy as np
import tensorflow as tf
import tensorflow.contrib.training as training

import sys
sys.path.append('/data/home/yiychen/Desktop/TSPerf/prototypes/retail_rnn_model/')
from utils import *
import hparams

# round number
ROUND = 1

# define parameters
# constant
predict_window = 3
is_train = False
mode = 'predict'
batch_size = 1024

# import hyper parameters
# TODO: add ema in the code to imporve the performance
hparams_dict = hparams.hparams_manual
hparams = training.HParams(**hparams_dict)

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

# read the saver from checkpoint
saver_path = os.path.join(intermediate_data_dir, 'cpt')
paths = [p for p in tf.train.get_checkpoint_state(saver_path).all_model_checkpoint_paths]

# build the dataset
root_ds = tf.data.Dataset.from_tensor_slices(
    (ts_value_train, feature_train, feature_test)).repeat()
batch = (root_ds
         .map(lambda *x: cut(*x, cut_mode=mode, train_window=hparams.train_window,
                             predict_window=predict_window, back_offset=0))
         .map(normalize_target)
         .batch(batch_size))
