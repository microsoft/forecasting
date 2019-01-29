# coding: utf-8

# Perform cross validation of a Dilated Convolutional Neural Network (CNN) model on the training data of the 1st forecast round. 

import os
import sys
import math
import keras
import argparse
import datetime
import numpy as np
import pandas as pd

from utils import *
from keras.layers import * 
from keras.models import Model
from keras import optimizers
from keras.utils import multi_gpu_model
from azureml.core import Run

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
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    parser.add_argument('--seq-len', type=int, dest='seq_len', default=20, help='length of the input sequence')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=64, help='mini batch size for training')
    parser.add_argument('--dropout-rate', type=float, dest='dropout_rate', default=0.10, help='dropout ratio')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.01, help='learning rate')
    parser.add_argument('--epochs', type=int, dest='epochs', default=30, help='# of epochs')
    args = parser.parse_args()
    args.dropout_rate = round(args.dropout_rate, 2)
    print(args)

    # Start an Azure ML run
    run = Run.get_context()

    # Data paths
    DATA_DIR = args.data_folder 
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')

    # Data and forecast problem parameters
    MAX_STORE_ID = 137
    MAX_BRAND_ID = 11
    PRED_HORIZON = 3
    PRED_STEPS = 2
    TRAIN_START_WEEK = 40
    TRAIN_END_WEEK_LIST = list(range(135,159,2))
    TEST_START_WEEK_LIST = list(range(137,161,2))
    TEST_END_WEEK_LIST = list(range(138,162,2))
    # The start datetime of the first week in the record
    FIRST_WEEK_START = pd.to_datetime('1989-09-14 00:00:00')

    # Input sequence length and feature names
    SEQ_LEN = args.seq_len 
    DYNAMIC_FEATURES = ['deal', 'feat', 'month', 'week_of_month', 'price', 'price_ratio']
    STATIC_FEATURES = ['store', 'brand']

    # Get unique stores and brands
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_round_1.csv'))
    store_list = train_df['store'].unique()
    brand_list = train_df['brand'].unique()
    store_brand = [(x,y) for x in store_list for y in brand_list] 

    # Train and validate the model using only the first round data
    r = 0
    print('---- Round ' + str(r+1) + ' ----')
    # Load training data
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_round_'+str(r+1)+'.csv'))
    train_df['move'] = train_df['logmove'].apply(lambda x: round(math.exp(x)))
    train_df = train_df[['store', 'brand', 'week', 'move']]

    # Create a dataframe to hold all necessary data
    week_list = range(TRAIN_START_WEEK, TEST_END_WEEK_LIST[r]+1)
    d = {'store': store_list,
         'brand': brand_list,
         'week': week_list}        
    data_grid = df_from_cartesian_product(d)
    data_filled = pd.merge(data_grid, train_df, how='left', 
                            on=['store', 'brand', 'week'])

    # Get future price, deal, and advertisement info
    aux_df = pd.read_csv(os.path.join(TRAIN_DIR, 'aux_round_'+str(r+1)+'.csv'))  
    data_filled = pd.merge(data_filled, aux_df, how='left',  
                            on=['store', 'brand', 'week'])

    # Create relative price feature
    price_cols = ['price1', 'price2', 'price3', 'price4', 'price5', 'price6', 'price7', 'price8', \
                  'price9', 'price10', 'price11']
    data_filled['price'] = data_filled.apply(lambda x: x.loc['price' + str(int(x.loc['brand']))], axis=1)
    data_filled['avg_price'] = data_filled[price_cols].sum(axis=1).apply(lambda x: x / len(price_cols))
    data_filled['price_ratio'] = data_filled.apply(lambda x: x['price'] / x['avg_price'], axis=1)

    # Fill missing values
    data_filled = data_filled.groupby(['store', 'brand']). \
                                apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    # Create datetime features
    data_filled['week_start'] = data_filled['week'].apply(lambda x: FIRST_WEEK_START + datetime.timedelta(days=(x-1)*7))
    data_filled['day'] = data_filled['week_start'].apply(lambda x: x.day)
    data_filled['week_of_month'] = data_filled['week_start'].apply(lambda x: week_of_month(x))
    data_filled['month'] = data_filled['week_start'].apply(lambda x: x.month)
    data_filled.drop('week_start', axis=1, inplace=True)  

    # Normalize the dataframe of features
    cols_normalize = data_filled.columns.difference(['store', 'brand', 'week'])
    data_scaled, min_max_scaler = normalize_dataframe(data_filled, cols_normalize)

    # Create sequence array for 'move'
    start_timestep = 0
    end_timestep = TRAIN_END_WEEK_LIST[r]-TRAIN_START_WEEK-PRED_HORIZON
    train_input1 = gen_sequence_array(data_scaled, store_brand, SEQ_LEN, ['move'], start_timestep, end_timestep)

    # Create sequence array for other dynamic features
    start_timestep = PRED_HORIZON
    end_timestep = TRAIN_END_WEEK_LIST[r]-TRAIN_START_WEEK
    train_input2 = gen_sequence_array(data_scaled, store_brand, SEQ_LEN, DYNAMIC_FEATURES, start_timestep, end_timestep)

    seq_in = np.concatenate((train_input1, train_input2), axis=2)

    # Create array of static features
    total_timesteps = TRAIN_END_WEEK_LIST[r]-TRAIN_START_WEEK-SEQ_LEN-PRED_HORIZON+2
    cat_fea_in = static_feature_array(data_filled, total_timesteps, STATIC_FEATURES)

    # Create training output
    start_timestep = SEQ_LEN+PRED_HORIZON-PRED_STEPS
    end_timestep = TRAIN_END_WEEK_LIST[r]-TRAIN_START_WEEK
    train_output = gen_sequence_array(data_filled, store_brand, PRED_STEPS, ['move'], start_timestep, end_timestep)
    train_output = np.squeeze(train_output)

    # Create model
    model = create_dcnn_model(seq_len=SEQ_LEN, n_filters=2, n_input_series=1+len(DYNAMIC_FEATURES), n_outputs=PRED_STEPS)

    # Convert to GPU model
    try:
        model = multi_gpu_model(model)
        print('Training using multiple GPUs...')
    except:
        print('Training using single GPU or CPU...')

    adam = optimizers.Adam(lr=args.learning_rate)
    model.compile(loss='mape', optimizer=adam, metrics=['mape', 'mae'])

    # Model training and validation
    history = model.fit([seq_in, cat_fea_in], train_output, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.05)
    val_loss = history.history['val_loss'][-1]
    print('Validation loss is {}'.format(val_loss))

    # Log the validation loss/MAPE
    run.log('MAPE', np.float(val_loss))


