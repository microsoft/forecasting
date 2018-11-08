# import packages
import os
import numpy as np
import tensorflow as tf
from utils import *

# round number
ROUND = 1

# define parameters
# constant
predict_window = 3
# tunable
train_window = 30

# read the numpy arrays output from the make_features.py
file_dir = './prototypes/retail_rnn_model'
# file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_relative_dir = '../../retail_sales/OrangeJuice_Pt_3Weeks_Weekly/data'
data_dir = os.path.join(file_dir, data_relative_dir)
intermediate_data_dir = os.path.join(data_dir, 'intermediate/round_{}'.format(ROUND))

ts_value_train = np.load(os.path.join(intermediate_data_dir, 'ts_value_train.npy'))
feature_train = np.load(os.path.join(intermediate_data_dir, 'feature_train.npy'))
feature_test = np.load(os.path.join(intermediate_data_dir, 'feature_test.npy'))

# make the dataset
root_ds = tf.data.Dataset.from_tensor_slices(
    (ts_value_train, feature_train, feature_test))\
    .repeat()\
    .map(lambda *x: cut(*x, cut_mode='train', back_offset=0))

# TODO: shuffle the time series
# TODO: filter the time series with too many zeros


# normalization on the target variable



root_ds_tmp = root_ds.map(lambda *x: cut(*x, cut_mode='eval'))
tensor0 = root_ds.make_one_shot_iterator().get_next()
tensor1 = root_ds_tmp.make_one_shot_iterator().get_next()
with tf.Session() as session:
    tmp0 = session.run(tensor0)
    tmp1 = session.run(tensor1)






