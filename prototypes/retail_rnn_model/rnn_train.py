import tensorflow as tf
import os
from utils import *
import numpy as np
import shutil

MODE = 'train'
IS_TRAIN = True


def rnn_train(ts_value_train, feature_train, feature_test, hparams, predict_window, intermediate_data_dir,
              submission_round, back_offset=0):

    # TODO: shuffle the time series
    # TODO: prefetch? optimization of perforamnce in time, n_threads in map etc.
    max_train_empty_percentage = 0.5
    max_train_empty = int(round(hparams.train_window * max_train_empty_percentage))

    # build the dataset
    root_ds = tf.data.Dataset.from_tensor_slices(
        (ts_value_train, feature_train, feature_test)).repeat()
    batch = (root_ds
             .map(lambda *x: cut(*x, cut_mode=MODE, train_window=hparams.train_window,
                                 predict_window=predict_window, ts_length=ts_value_train.shape[1], back_offset=back_offset))
             .filter(lambda *x: reject_filter(max_train_empty, *x))
             .map(normalize_target)
             .batch(hparams.batch_size))

    iterator = batch.make_initializable_iterator()
    it_tensors = iterator.get_next()
    true_x, true_y, feature_x, feature_y, norm_x, norm_mean, norm_std = it_tensors
    encoder_feature_depth = feature_x.shape[2].value

    # build the model, get the predictions
    predictions = build_rnn_model(norm_x, feature_x, feature_y, norm_mean, norm_std, predict_window, IS_TRAIN, hparams)

    # calculate loss on log scale
    mae_loss = calc_mae_loss(true_y, predictions)
    # calculate differntiable mape loss on original scale, this is the metric to be optimized
    mape_loss = calc_differentiable_mape_loss(true_y, predictions)
    # calculate rounded mape on original scale, this is the metric which is identitial to the final evaluation metric
    mape = calc_rounded_mape(true_y, predictions)

    # Sum all losses
    total_loss = mape_loss
    train_op, glob_norm, ema = make_train_op(total_loss, hparams.learning_rate,  hparams.beta1, hparams.beta2,
                                             hparams.epsilon, hparams.asgd_decay)

    train_size = ts_value_train.shape[0]
    steps_per_epoch = train_size // hparams.batch_size

    global_step = tf.Variable(0, name='global_step', trainable=False)
    inc_step = tf.assign_add(global_step, 1)

    saver = tf.train.Saver(max_to_keep=1, name='train_saver')
    init = tf.global_variables_initializer()

    results_mae = []
    results_mape = []
    results_mape_loss = []

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          gpu_options=tf.GPUOptions(allow_growth=False))) as sess:
        sess.run(init)
        sess.run(iterator.initializer)

        for epoch in range(hparams.max_epoch):
            results_epoch_mae = []
            results_epoch_mape = []
            results_epoch_mape_loss = []

            tqr = range(steps_per_epoch)

            for _ in tqr:
                try:
                    ops = [inc_step]
                    ops.extend([train_op])
                    ops.extend([mae_loss, mape, mape_loss, glob_norm])

                    # for debug
                    # ops.extend([predictions, true_x, true_y, feature_x, feature_y, norm_x, norm_mean, norm_std])

                    results = sess.run(ops)

                    # get the results
                    step = results[0]

                    step_mae = results[2]
                    step_mape = results[3]
                    step_mape_loss = results[4]

                    # for debug
                    # step_predictions = results[6]
                    # step_true_x = results[7]
                    # step_true_y = results[8]
                    # step_feature_x = results[9]
                    # step_feature_y = results[10]
                    # step_norm_x = results[11]
                    # step_norm_mean = results[12]
                    # step_norm_std = results[13]

                    print(
                        'step: {}, MAE: {}, MAPE: {}, MAPE_LOSS: {}'.format(step, step_mae, step_mape, step_mape_loss))

                    results_epoch_mae.append(step_mae)
                    results_epoch_mape.append(step_mape)
                    results_epoch_mape_loss.append(step_mape_loss)

                except tf.errors.OutOfRangeError:
                    break

            # append the results
            results_mae.append(results_epoch_mae)
            results_mape.append(results_epoch_mape)
            results_mape_loss.append(results_epoch_mape_loss)

        step = results[0]
        saver_path = os.path.join(intermediate_data_dir, 'cpt_round_{}'.format(submission_round))
        if os.path.exists(saver_path):
            shutil.rmtree(saver_path)
        saver.save(sess, os.path.join(saver_path, 'cpt'), global_step=step, write_state=True)

    # look at the training results
    # examine step_mae and step_mape_loss
    print('MAE in epochs')
    print(np.mean(results_mae, axis=1))
    print('MAPE LOSS in epochs')
    print(np.mean(results_mape_loss, axis=1))
    print('MAPE in epochs')
    print(np.mean(results_mape, axis=1))

    return np.mean(results_mape, axis=1)[-1]


