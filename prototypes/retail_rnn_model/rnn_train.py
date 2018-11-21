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
import hparams

# round number
ROUND = 1

# define parameters
# constant
predict_window = 3
is_train = True
mode = 'train'
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

# TODO: shuffle the time series
# TODO: filter the time series with too many zeros
# TODO: prefetch? optimization of perforamnce in time, n_threads in map etc.
# build the dataset
root_ds = tf.data.Dataset.from_tensor_slices(
    (ts_value_train, feature_train, feature_test)).repeat()
batch = (root_ds
         .map(lambda *x: cut(*x, cut_mode=mode, train_window=hparams.train_window,
                             predict_window=predict_window, ts_length=ts_value_train.shape[1], back_offset=0))
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
mask = tf.logical_not(tf.math.equal(true_y, tf.zeros_like(true_y)))
# Fill NaNs by zeros (can use any value)
# true_y = tf.where(mask, true_y, tf.zeros_like(true_y))
# Assign zero weight to NaNs
weights = tf.to_float(mask)
mae_loss = tf.losses.absolute_difference(labels=true_y, predictions=predictions, weights=weights)

# mape_loss
epsilon = 0.1  # Smoothing factor, helps SMAPE to be well-behaved near zero
true_o = tf.expm1(true_y)
pred_o = tf.expm1(predictions)
# summ = tf.maximum(tf.abs(true_o) + epsilon, 0.5 + epsilon)
mape_loss_origin = tf.abs(pred_o - true_o) / (tf.abs(true_o) + epsilon)
mape_loss = tf.losses.compute_weighted_loss(mape_loss_origin, weights, loss_collection=None)

# mape
true_o1 = tf.round(tf.expm1(true_y))
pred_o1 = tf.maximum(tf.round(tf.expm1(predictions)), 0.0)
raw_mape = tf.abs(pred_o1 - true_o1) / tf.abs(true_o1)
raw_mape_mask = tf.is_finite(raw_mape)
raw_mape_weights = tf.to_float(raw_mape_mask)
raw_mape_filled = tf.where(raw_mape_mask, raw_mape, tf.zeros_like(raw_mape))
mape = tf.losses.compute_weighted_loss(raw_mape_filled, raw_mape_weights, loss_collection=None)

# mae, mape_loss, mape, loss_item_count = calc_loss(predictions, true_y)
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

results_mae = []
results_mape = []
results_mape_loss = []
results_weights = []
results_true_y = []
results_predictions = []
results_true_o = []
results_pred_o = []
results_true_o1 = []
results_pred_o1 = []

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                      gpu_options=tf.GPUOptions(allow_growth=False))) as sess:
    sess.run(init)
    sess.run(iterator.initializer)

    for epoch in range(hparams.max_epoch):
        results_epoch_mae = []
        results_epoch_mape = []
        results_epoch_mape_loss = []
        results_epoch_weights = []
        results_epoch_true_y = []
        results_epoch_predictions = []
        results_epoch_true_o = []
        results_epoch_pred_o = []
        results_epoch_true_o1 = []
        results_epoch_pred_o1 = []

        tqr = range(steps_per_epoch)

        for _ in tqr:
            try:
                ops = [inc_step]
                ops.extend([train_op])
                ops.extend([mae_loss, mape, mape_loss, glob_norm])
                ops.extend([weights, true_y, predictions, true_o, pred_o, true_o1, pred_o1])
                results = sess.run(ops)

                # get the results
                step = results[0]

                step_mae = results[2]
                step_mape = results[3]
                step_mape_loss = results[4]
                step_weights = results[6]
                step_true_y = results[7]
                step_predictions = results[8]
                step_true_o = results[9]
                step_pred_o = results[10]
                step_true_o1 = results[11]
                step_pred_o1 = results[12]

                print('step: {}, MAE: {}, MAPE: {}, MAPE_LOSS: {}'.format(step, step_mae, step_mape, step_mape_loss))

                results_epoch_mae.append(step_mae)
                results_epoch_mape.append(step_mape)
                results_epoch_mape_loss.append(step_mape_loss)
                results_epoch_weights.append(step_weights)
                results_epoch_true_y.append(step_true_y)
                results_epoch_predictions.append(step_predictions)
                results_epoch_true_o.append(step_true_o)
                results_epoch_pred_o.append(step_pred_o)
                results_epoch_true_o1.append(step_true_o1)
                results_epoch_pred_o1.append(step_pred_o1)

            except tf.errors.OutOfRangeError:
                break


        # append the results
        results_mae.append(results_epoch_mae)
        results_mape.append(results_epoch_mape)
        results_mape_loss.append(results_epoch_mape_loss)
        results_weights.append(results_epoch_weights)
        results_true_y.append(results_epoch_true_y)
        results_predictions.append(results_epoch_predictions)
        results_true_o.append(results_epoch_true_o)
        results_pred_o.append(results_epoch_pred_o)
        results_true_o1.append(results_epoch_true_o1)
        results_pred_o1.append(results_epoch_pred_o1)

    step = results[0]
    saver_path = os.path.join(intermediate_data_dir, 'cpt')
    saver.save(sess, saver_path, global_step=step)


# look at the training results
# examine step_mae and step_mape_loss
print('MAPE in pochs')
print(np.mean(results_mape, axis=1))
print('MAE in pochs')
print(np.mean(results_mae, axis=1))


