"""
Revise based on: https://github.com/Arturus/kaggle-web-traffic/blob/master/model.py
"""

# import packages
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.cudnn_rnn as cudnn_rnn
from tensorflow.python.util import nest

import sys
sys.path.append('C:\\Users\\yiychen\\Desktop\\repos\\TSPerf\\prototypes\\retail_rnn_model')
from utils import *

# round number
ROUND = 1

# define parameters
# constant
predict_window = 3
is_train = True
# tunable
train_window = 30
batch_size = 64

# read the numpy arrays output from the make_features.py
file_dir = './prototypes/retail_rnn_model'
# file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
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
         .map(lambda *x: cut(*x, cut_mode='train', train_window=train_window,
                             predict_window=predict_window, back_offset=0))
         .map(normalize_target)
         .batch(batch_size))

iterator = batch.make_initializable_iterator()
it_tensors = iterator.get_next()
true_x, true_y, feature_x, feature_y, norm_x, norm_mean, norm_std = it_tensors
encoder_feature_depth = feature_x.shape[2].value

# build the encoder-decoder RNN model
RNN = cudnn_rnn.CudnnGRU
hparams = {}
hparams['encoder_rnn_layers'] = 1
hparams['decoder_rnn_layers'] = hparams['encoder_rnn_layers']
hparams['rnn_depth'] = 200
hparams['encoder_dropout'] = 0.03


def make_encoder(time_inputs, is_train, hparams):
    """
    Builds encoder, using CUDA RNN
    """

    def build_rnn():
        return RNN(num_layers=hparams['encoder_rnn_layers'], num_units=hparams['rnn_depth'],
                   kernel_initializer=tf.initializers.random_uniform(minval=-0.05, maxval=0.05),
                   direction='unidirectional',
                   dropout=hparams['encoder_dropout'] if is_train else 0)

    cuda_model = build_rnn()

    # [batch, time, features] -> [time, batch, features]
    time_first = tf.transpose(time_inputs, [1, 0, 2])
    rnn_time_input = time_first
    # rnn_out: (time, batch, rnn_depth)
    # rnn_state: (num_layers, batch, rnn_depth)
    rnn_out, (rnn_state,) = cuda_model(inputs=rnn_time_input)
    return rnn_out, rnn_state


encoder_output, h_state = make_encoder(feature_x, is_train, hparams)


# apply dropout to the h_state during training time
def wrap_dropout(structure, dropout):
    if dropout < 1.0:
        return nest.map_structure(lambda x: tf.nn.dropout(x, keep_prob=dropout), structure)
    else:
        return structure


def convert_cudnn_state_v2(h_state, hparams, seed, c_state=None, dropout=1.0):
    """
    Converts RNN state tensor from cuDNN representation to TF RNNCell compatible representation.
    :param h_state: tensor [num_layers, batch_size, depth]
    :param c_state: LSTM additional state, should be same shape as h_state
    :return: TF cell representation matching RNNCell.state_size structure for compatible cell
    """

    def squeeze(seq):
        return tuple(seq) if len(seq) > 1 else seq[0]

    def wrap_dropout(structure):
        if dropout < 1.0:
            return nest.map_structure(lambda x: tf.nn.dropout(x, keep_prob=dropout, seed=seed), structure)
        else:
            return structure

    # Cases:
    # decoder_layer = encoder_layers, straight mapping
    # encoder_layers > decoder_layers: get outputs of upper encoder layers
    # encoder_layers < decoder_layers: feed encoder outputs to lower decoder layers, feed zeros to top layers
    h_layers = tf.unstack(h_state)
    if hparams.encoder_rnn_layers >= hparams.decoder_rnn_layers:
        return squeeze(wrap_dropout(h_layers[hparams.encoder_rnn_layers - hparams.decoder_rnn_layers:]))
    else:
        lower_inputs = wrap_dropout(h_layers)
        upper_inputs = [tf.zeros_like(h_layers[0]) for _ in
                        range(hparams.decoder_rnn_layers - hparams.encoder_rnn_layers)]
        return squeeze(lower_inputs + upper_inputs)

encoder_state = convert_cudnn_state_v2(h_state, hparams,
                                               dropout=hparams.gate_dropout if is_train else 1.0)