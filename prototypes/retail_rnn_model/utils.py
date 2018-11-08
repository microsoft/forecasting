import tensorflow as tf


def cut(ts_value_train_slice, feature_train_slice,
        feature_test_slice, train_window, predict_window,
        cut_mode='train', back_offset=0):
    """
    cut each element of the dataset into x and y for supervised learning.

    :param ts_value_train_slice: shape of (#train_ts_length,)
    :param feature_train_slice: shape of (#train_ts_length, #features)
    :param feature_test_slice: shape of (#test_ts_length, #features)
    :param cut_mode: 'train', 'eval', 'predict'
    :param back_offset: how many data points at end of time series
            cannot be used for training.
            set back_offset = predict_window for training
            during hyper parameter tuning.

    :return:
    """
    ts_length = tf.shape(ts_value_train_slice)[0]
    if cut_mode in ['train', 'eval']:
        if cut_mode == 'train':
            min_start_idx = 0
            max_start_idx = (ts_length - back_offset) - \
                            (train_window + predict_window) + 1
            train_start = tf.random_uniform((), min_start_idx, max_start_idx,
                                            dtype=tf.int32)
        elif cut_mode == 'eval':
            train_start = ts_length - (train_window + predict_window)

        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + predict_window

        true_x = ts_value_train_slice[train_start: train_end]
        true_y = ts_value_train_slice[test_start: test_end]
        feature_x = feature_train_slice[train_start: train_end]
        feature_y = feature_train_slice[test_start: test_end]

    else:
        train_start = ts_length - train_window
        train_end = train_start + train_window

        true_x = ts_value_train_slice[train_start: train_end]
        true_y = None
        feature_x = feature_train_slice[train_start: train_end]
        feature_y = feature_test_slice

    return true_x, true_y, feature_x, feature_y
