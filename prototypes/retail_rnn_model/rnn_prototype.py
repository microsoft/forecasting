"""
Revise based on: https://github.com/Arturus/kaggle-web-traffic/blob/master/model.py
"""

# import packages
import os
import inspect
import numpy as np
import tensorflow as tf
import tensorflow.contrib.training as training

import sys
sys.path.append('/data/home/yiychen/Desktop/TSPerf/prototypes/retail_rnn_model/')
from utils import *

# round number
ROUND = 1

# define parameters
# constant
predict_window = 3
is_train = True
mode = 'train'
# tunable
hparams_dict = {}
hparams_dict['train_window'] = 30
hparams_dict['batch_size'] = 64
hparams_dict['encoder_rnn_layers'] = 1
hparams_dict['decoder_rnn_layers'] = hparams_dict['encoder_rnn_layers']
hparams_dict['rnn_depth'] = 200
hparams_dict['encoder_dropout'] = 0.03
hparams_dict['gate_dropout'] = 0.997
hparams_dict['decoder_input_dropout'] = [1.0]
hparams_dict['decoder_state_dropout'] = [0.99]
hparams_dict['decoder_output_dropout'] = [0.975]
hparams_dict['decoder_variational_dropout'] = [False]
hparams_dict['asgd_decay'] = None
# TODO: add ema in the code to imporve the performance
# hparams_dict['asgd_decay'] = 0.99
hparams_dict['max_epoch'] = 20

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

# TODO: shuffle the time series
# TODO: filter the time series with too many zeros
# TODO: prefetch? optimization of perforamnce in time, n_threads in map etc.
# build the dataset
root_ds = tf.data.Dataset.from_tensor_slices(
    (ts_value_train, feature_train, feature_test)).repeat()
batch = (root_ds
         .map(lambda *x: cut(*x, cut_mode=mode, train_window=hparams.train_window,
                             predict_window=predict_window, back_offset=0))
         .map(normalize_target)
         .batch(hparams.batch_size))

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

# calculate loss
if mode == 'predict':
    # [Yiyu]: not sure why need this?
    # Pseudo-apply ema to get variable names later in ema.variables_to_restore()
    # This is copypaste from make_train_op()
    if hparams.asgd_decay:
        ema = tf.train.ExponentialMovingAverage(decay=hparams.asgd_decay)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        ema_vars = variables
        ema.apply(ema_vars)
else:
    mae, mape_loss, mape, loss_item_count = calc_loss(predictions, true_y)
    if is_train:
        # Sum all losses
        total_loss = mape_loss
        train_op, glob_norm, ema = make_train_op(total_loss, hparams.asgd_decay)

train_size = ts_value_train.shape[0]
steps_per_epoch = train_size // hparams.batch_size


# global_step = tf.train.get_or_create_global_step()
global_step = tf.Variable(0, name='global_step', trainable=False)
inc_step = tf.assign_add(global_step, 1)

saver = tf.train.Saver(max_to_keep=1, name='train_saver')
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                      gpu_options=tf.GPUOptions(allow_growth=False))) as sess:
    sess.run(init)
    sess.run(iterator.initializer)

    for epoch in range(hparams.max_epoch):
        tqr = range(steps_per_epoch)

        for _ in tqr:
            try:
                ops = [inc_step]
                ops.extend([train_op])
                ops.extend([mae, mape, glob_norm])
                results = sess.run(ops)
            except tf.errors.OutOfRangeError:
                break

    step = results[0]
    saver_path = os.path.join(intermediate_data_dir, 'cpt')
    saver.save(sess, saver_path, global_step=step)

