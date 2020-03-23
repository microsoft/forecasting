"""
This script contains the function for creating predictions for the RNN model.
"""

# import packages
import os
import numpy as np
import tensorflow as tf

from utils import *

# define parameters
IS_TRAIN = False


def rnn_predict(
    ts_value_train,
    feature_train,
    feature_test,
    hparams,
    predict_window,
    intermediate_data_dir,
    submission_round,
    batch_size,
    cut_mode="predict",
):
    """
    This function creates predictions by loading the trained RNN model.

    Args:
        ts_value_train: Numpy array which contains the time series value in the
            training dataset in shape of (#time series, #train_ts_length)
        feature_train: Numpy array which contains the feature values in the
            training dataset in shape of (#time series, #train_ts_length,
            #features)
        feature_test: Numpy array which contains the feature values for the
            test dataset in shape of (#time series, #test_ts_length)
        hparams: the tensorflow HParams object which contains the
            hyperparameter of the RNN model.
        predict_window: Integer, predict horizon.
        intermediate_data_dir: String, the directory which stores the
            intermediate results.
        submission_round: Integer, the submission round.
        batch_size: Integer, the batch size for making RNN predictions.
        cut_mode: 'train', 'eval' or 'predict'.
    Returns:
        pred_o: Numpy array which contains the predictions in shape of
        (#time series, #predict_window)
    """
    # build the dataset
    root_ds = tf.data.Dataset.from_tensor_slices((ts_value_train, feature_train, feature_test)).repeat(1)
    batch = (
        root_ds.map(
            lambda *x: cut(
                *x,
                cut_mode=cut_mode,
                train_window=hparams.train_window,
                predict_window=predict_window,
                ts_length=ts_value_train.shape[1],
                back_offset=0
            )
        )
        .map(normalize_target)
        .batch(batch_size)
    )

    iterator = batch.make_initializable_iterator()
    it_tensors = iterator.get_next()
    true_x, true_y, feature_x, feature_y, norm_x, norm_mean, norm_std = it_tensors

    # build the model, get the predictions
    predictions = build_rnn_model(norm_x, feature_x, feature_y, norm_mean, norm_std, predict_window, IS_TRAIN, hparams)

    # init the saver
    saver = tf.train.Saver(name="eval_saver", var_list=None)
    # read the saver from checkpoint
    saver_path = os.path.join(intermediate_data_dir, "cpt_round_{}".format(submission_round))
    paths = [p for p in tf.train.get_checkpoint_state(saver_path).all_model_checkpoint_paths]
    checkpoint = paths[0]

    # run the session
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(iterator.initializer)
        saver.restore(sess, checkpoint)

        (pred,) = sess.run([predictions])

    # invert the prediction back to original scale
    pred_o = np.exp(pred) - 1
    pred_o = pred_o.astype(int)

    return pred_o
