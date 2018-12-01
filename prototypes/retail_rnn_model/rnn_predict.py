# import packages
import os
import numpy as np
import tensorflow as tf

from utils import *

# define parameters
MODE = 'predict'
IS_TRAIN = False


def rnn_predict(ts_value_train, feature_train, feature_test, hparams, predict_window, intermediate_data_dir,
                submission_round, batch_size):
    # build the dataset
    root_ds = tf.data.Dataset.from_tensor_slices(
        (ts_value_train, feature_train, feature_test)).repeat(1)
    batch = (root_ds
             .map(lambda *x: cut(*x, cut_mode=MODE, train_window=hparams.train_window,
                                 predict_window=predict_window, ts_length=ts_value_train.shape[1], back_offset=0))
             .map(normalize_target)
             .batch(batch_size))

    iterator = batch.make_initializable_iterator()
    it_tensors = iterator.get_next()
    true_x, true_y, feature_x, feature_y, norm_x, norm_mean, norm_std = it_tensors
    encoder_feature_depth = feature_x.shape[2].value

    # build the model, get the predictions
    predictions = build_rnn_model(norm_x, feature_x, feature_y, norm_mean, norm_std, predict_window, IS_TRAIN, hparams)

    # init the saver
    saver = tf.train.Saver(name='eval_saver', var_list=None)
    # read the saver from checkpoint
    saver_path = os.path.join(intermediate_data_dir, 'cpt_round_{}'.format(submission_round))
    paths = [p for p in tf.train.get_checkpoint_state(saver_path).all_model_checkpoint_paths]
    checkpoint = paths[0]

    # run the session
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(iterator.initializer)
        saver.restore(sess, checkpoint)

        pred, = sess.run([predictions])

    # invert the prediction back to original scale
    pred_o = np.exp(pred)
    return pred_o


