# import packages
import os
import inspect
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.training as training

import sys
sys.path.append('/data/home/yiychen/Desktop/TSPerf/prototypes/retail_rnn_model/')
from utils import *
import hparams
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

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

# build the dataset
root_ds = tf.data.Dataset.from_tensor_slices(
    (ts_value_train, feature_train, feature_test)).repeat(1)
batch = (root_ds
         .map(lambda *x: cut(*x, cut_mode=mode, train_window=hparams.train_window,
                             predict_window=predict_window, ts_length=ts_value_train.shape[1], back_offset=0))
         .map(normalize_target)
         .batch(batch_size))

iterator = batch.make_initializable_iterator()
it_tensors = iterator.get_next()
true_x, true_y, feature_x, feature_y, norm_x, norm_mean, norm_std = it_tensors
encoder_feature_depth = feature_x.shape[2].value

# build the encoder-decoder RNN model
# make encoder
x_all_features = tf.concat([tf.expand_dims(norm_x, -1), feature_x], axis=-1)
encoder_output, h_state = make_encoder(x_all_features, is_train, hparams)


encoder_state = convert_cudnn_state_v2(h_state, hparams,
                                       dropout=hparams.gate_dropout if is_train else 1.0)

# Run decoder
decoder_targets, decoder_outputs = decoder(encoder_state, feature_y, norm_x[:, -1], hparams, is_train=is_train,
                                           predict_window=predict_window)

# get predictions
predictions = decode_predictions(decoder_targets, norm_mean, norm_std)

# init the saver
saver = tf.train.Saver(name='eval_saver', var_list=None)
# read the saver from checkpoint
saver_path = os.path.join(intermediate_data_dir, 'cpt')
paths = [p for p in tf.train.get_checkpoint_state(saver_path).all_model_checkpoint_paths]
checkpoint = paths[0]

# run the session
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    sess.run(iterator.initializer)
    saver.restore(sess, checkpoint)

    pred, = sess.run([predictions])

# arrange the predictions into pd.DataFrame
pred_o = np.exp(pred)
pred_sub = pred_o[:, 1:].reshape((-1))

# read in the test_file for this round
test_file = os.path.join(data_dir, 'test/test_round_{}.csv'.format(ROUND))
test = pd.read_csv(test_file, index_col=False)
submission = test.sort_values(by=['store', 'brand', 'week'], ascending=True)[['store', 'brand', 'week', 'logmove']]
submission['round'] = ROUND
submission['weeks_ahead'] = submission['week'] - bs.TRAIN_END_WEEK_LIST[ROUND - 1]
submission['prediction'] = pred_sub
submission = submission[['round', 'store', 'brand', 'week', 'weeks_ahead', 'prediction', 'logmove']]

# write the submission
submission['move'] = np.exp(submission['logmove'])

from common.metrics import MAPE
print("MAPE: ", MAPE(submission['prediction'], submission['move'])*100)

