# coding: utf-8

# Train and score a Dilated Convolutional Neural Network (CNN) model using Keras package with TensorFlow backend. 

import os
import sys
import keras
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import optimizers
from keras.layers import * 
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint

# Append TSPerf path to sys.path (assume we run the script from TSPerf directory)
tsperf_dir = '.' 
if tsperf_dir not in sys.path:
    sys.path.append(tsperf_dir)

# Import TSPerf components
from utils import *
from make_features import make_features
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

# Model definition
def create_dcnn_model(seq_len, kernel_size=2, n_filters=3, n_input_series=1, n_outputs=1):
    """Create a Dilated CNN model.

    Args: 
        seq_len (Integer): Input sequence length
        kernel_size (Integer): Kernel size of each convolutional layer
        n_filters (Integer): Number of filters in each convolutional layer
        n_outputs (Integer): Number of outputs in the last layer

    Returns:
        Keras Model object
    """
    # Sequential input
    seq_in = Input(shape=(seq_len, n_input_series))

    # Categorical input
    cat_fea_in = Input(shape=(2,), dtype='uint8')
    store_id = Lambda(lambda x: x[:, 0, None])(cat_fea_in)
    brand_id = Lambda(lambda x: x[:, 1, None])(cat_fea_in)
    store_embed = Embedding(MAX_STORE_ID+1, 7, input_length=1)(store_id)
    brand_embed = Embedding(MAX_BRAND_ID+1, 4, input_length=1)(brand_id)

    # Dilated convolutional layers
    c1 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=1, 
                padding='causal', activation='relu')(seq_in)
    c2 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=2, 
                padding='causal', activation='relu')(c1)
    c3 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=4, 
                padding='causal', activation='relu')(c2)

    # Skip connections
    c4 = concatenate([c1, c3])

    # Output of convolutional layers 
    conv_out = Conv1D(8, 1, activation='relu')(c4)
    conv_out = Dropout(args.dropout_rate)(conv_out) 
    conv_out = Flatten()(conv_out)
    
    # Concatenate with categorical features
    x = concatenate([conv_out, Flatten()(store_embed), Flatten()(brand_embed)])
    x = Dense(16, activation='relu')(x) 
    output = Dense(n_outputs, activation='linear')(x)
    
    # Define model interface, loss function, and optimizer
    model = Model(inputs=[seq_in, cat_fea_in], outputs=output)

    return model

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, dest='seed', default=1, help='random seed')
    parser.add_argument('--seq-len', type=int, dest='seq_len', default=15, help='length of the input sequence')
    parser.add_argument('--dropout-rate', type=float, dest='dropout_rate', default=0.01, help='dropout ratio')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=64, help='mini batch size for training')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.015, help='learning rate')
    parser.add_argument('--epochs', type=int, dest='epochs', default=25, help='# of epochs')
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # Data paths
    DATA_DIR = os.path.join(tsperf_dir, 'retail_sales', 'OrangeJuice_Pt_3Weeks_Weekly', 'data') 
    SUBMISSION_DIR = os.path.join(tsperf_dir, 'retail_sales', 'OrangeJuice_Pt_3Weeks_Weekly', 'submissions', 'DilatedCNN') 
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')

    # Dataset parameters
    MAX_STORE_ID = 137
    MAX_BRAND_ID = 11

    # Parameters of the model
    PRED_HORIZON = 3
    PRED_STEPS = 2
    SEQ_LEN = args.seq_len 
    DYNAMIC_FEATURES = ['deal', 'feat', 'month', 'week_of_month', 'price', 'price_ratio']  
    STATIC_FEATURES = ['store', 'brand']

    # Get unique stores and brands
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_round_1.csv'))
    store_list = train_df['store'].unique()
    brand_list = train_df['brand'].unique()
    store_brand = [(x,y) for x in store_list for y in brand_list] 

    # Train and predict for all forecast rounds
    pred_all = []
    file_name = os.path.join(SUBMISSION_DIR, 'dcnn_model.h5')
    for r in range(bs.NUM_ROUNDS):
        print('---- Round ' + str(r+1) + ' ----')
        offset = 0 if r==0 else 40+r*PRED_STEPS 
        # Create features 
        data_filled, data_scaled = make_features(r, TRAIN_DIR, PRED_STEPS, offset, store_list, brand_list)

        # Create sequence array for 'move'
        start_timestep = 0
        end_timestep = bs.TRAIN_END_WEEK_LIST[r]-bs.TRAIN_START_WEEK-PRED_HORIZON
        train_input1 = gen_sequence_array(data_scaled, store_brand, SEQ_LEN, ['move'], start_timestep, end_timestep-offset)

        # Create sequence array for other dynamic features
        start_timestep = PRED_HORIZON
        end_timestep = bs.TRAIN_END_WEEK_LIST[r]-bs.TRAIN_START_WEEK
        train_input2 = gen_sequence_array(data_scaled, store_brand, SEQ_LEN, DYNAMIC_FEATURES, start_timestep, end_timestep-offset)

        seq_in = np.concatenate([train_input1, train_input2], axis=2)

        # Create array of static features
        total_timesteps = bs.TRAIN_END_WEEK_LIST[r]-bs.TRAIN_START_WEEK-SEQ_LEN-PRED_HORIZON+2
        cat_fea_in = static_feature_array(data_filled, total_timesteps-offset, STATIC_FEATURES)

        # Create training output
        start_timestep = SEQ_LEN+PRED_HORIZON-PRED_STEPS
        end_timestep = bs.TRAIN_END_WEEK_LIST[r]-bs.TRAIN_START_WEEK
        train_output = gen_sequence_array(data_filled, store_brand, PRED_STEPS, ['move'], start_timestep, end_timestep-offset)
        train_output = np.squeeze(train_output)

        # Create and train model
        if r == 0:
            model = create_dcnn_model(seq_len=SEQ_LEN, n_filters=2, n_input_series=1+len(DYNAMIC_FEATURES), n_outputs=PRED_STEPS)
            adam = optimizers.Adam(lr=args.learning_rate)
            model.compile(loss='mape', optimizer=adam, metrics=['mape'])
            # Define checkpoint and fit model
            checkpoint = ModelCheckpoint(file_name, monitor='loss', save_best_only=True, mode='min', verbose=0)
            callbacks_list = [checkpoint]
            history = model.fit([seq_in, cat_fea_in], train_output, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks_list, verbose=0)
        else:
            model = load_model(file_name)
            checkpoint = ModelCheckpoint(file_name, monitor='loss', save_best_only=True, mode='min', verbose=0)
            callbacks_list = [checkpoint]
            history = model.fit([seq_in, cat_fea_in], train_output, epochs=1, batch_size=args.batch_size, callbacks=callbacks_list, verbose=0)        

        # Get inputs for prediction
        start_timestep = bs.TEST_START_WEEK_LIST[r] - bs.TRAIN_START_WEEK - SEQ_LEN - PRED_HORIZON + PRED_STEPS
        end_timestep = bs.TEST_START_WEEK_LIST[r] - bs.TRAIN_START_WEEK + PRED_STEPS - 1 - PRED_HORIZON
        test_input1 = gen_sequence_array(data_scaled, store_brand, SEQ_LEN, ['move'], start_timestep-offset, end_timestep-offset)

        start_timestep = bs.TEST_END_WEEK_LIST[r] - bs.TRAIN_START_WEEK - SEQ_LEN + 1
        end_timestep = bs.TEST_END_WEEK_LIST[r] - bs.TRAIN_START_WEEK
        test_input2 = gen_sequence_array(data_scaled, store_brand, SEQ_LEN, DYNAMIC_FEATURES, start_timestep-offset, end_timestep-offset)

        seq_in = np.concatenate([test_input1, test_input2], axis=2)

        total_timesteps = 1
        cat_fea_in = static_feature_array(data_filled, total_timesteps, STATIC_FEATURES)

        # Make prediction
        pred = np.round(model.predict([seq_in, cat_fea_in]))
        
        # Create dataframe for submission
        exp_output = data_filled[data_filled.week >= bs.TEST_START_WEEK_LIST[r]].reset_index(drop=True)
        exp_output = exp_output[['store', 'brand', 'week']]
        pred_df = exp_output.sort_values(['store', 'brand', 'week']).\
                             loc[:,['store', 'brand', 'week']].\
                             reset_index(drop=True)
        pred_df['weeks_ahead'] = pred_df['week'] - bs.TRAIN_END_WEEK_LIST[r]
        pred_df['round'] = r+1
        pred_df['prediction'] = np.reshape(pred, (pred.size, 1))
        pred_all.append(pred_df)

    # Generate submission
    submission = pd.concat(pred_all, axis=0).reset_index(drop=True)
    submission = submission[['round', 'store', 'brand', 'week', 'weeks_ahead', 'prediction']]
    filename = 'submission_seed_' + str(args.seed) + '.csv'
    submission.to_csv(os.path.join(SUBMISSION_DIR, filename), index=False)
    print('Done')


    

