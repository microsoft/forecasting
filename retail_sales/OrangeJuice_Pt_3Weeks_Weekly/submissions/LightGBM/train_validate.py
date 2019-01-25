# coding: utf-8

# Perform cross validation of a boosted decision tree model on the training data of the 1st forecast round. 

import os
import sys
import math
import argparse
import datetime
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb 
from azureml.core import Run
from sklearn.model_selection import train_test_split

# Utility functions
def week_of_month(dt):
    """Get the week of the month for the specified date.
    
    Args: 
        dt (Datetime): Input date
        
    Returns:
        wom (Integer): Week of the month of the input date
    """ 
    from math import ceil
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    wom = int(ceil(adjusted_dom/7.0))
    return wom

def df_from_cartesian_product(dict_in):
    """Generate a Pandas dataframe from Cartesian product of lists.
    
    Args: 
        dict_in (Dictionary): Dictionary containing multiple lists
        
    Returns:
        df (Dataframe): Dataframe corresponding to the Caresian product of the lists
    """
    from collections import OrderedDict
    from itertools import product
    od = OrderedDict(sorted(dict_in.items()))
    cart = list(product(*od.values()))
    df = pd.DataFrame(cart, columns=od.keys())
    return df

def lagged_features(df, lags):
    """Create lagged features based on time series data.
    
    Args:
        df (Dataframe): Input time series data sorted by time
        lags (List): Lag lengths
        
    Returns:
        fea (Dataframe): Lagged features 
    """
    df_list = []
    for lag in lags:
        df_shifted = df.shift(lag)
        df_shifted.columns = [x + '_lag' + str(lag) for x in df_shifted.columns]
        df_list.append(df_shifted)
    fea = pd.concat(df_list, axis=1)
    return fea

def moving_averages(df, start_step, window_size=None):
    """Compute averages of every feature over moving time windows.
    
    Args:
        df (Dataframe): Input features as a dataframe
        start_step (Integer): Starting time step of rolling mean
        window_size (Integer): Windows size of rolling mean
        
    Returns:
        fea (Dataframe): Dataframe consisting of the moving averages
    """
    if window_size == None: # Use a large window to compute average over all historical data
        window_size = df.shape[0]
    fea = df.shift(start_step).rolling(min_periods=1, center=False, window=window_size).mean()
    fea.columns = fea.columns + '_mean'
    return fea

def combine_features(df, lag_fea, lags, window_size, used_columns):
    """Combine different features for a certain store-brand.
    
    Args:
        df (Dataframe): Time series data of a certain store-brand
        lag_fea (List): A list of column names for creating lagged features
        lags (Numpy Array): Numpy array including all the lags
        window_size (Integer): Windows size of rolling mean
        used_columns (List): A list of names of columns used in model training (including target variable)
    
    Returns:
        fea_all (Dataframe): Dataframe including all features for the specific store-brand
    """
    lagged_fea = lagged_features(df[lag_fea], lags)
    moving_avg = moving_averages(df[lag_fea], 2, window_size)
    fea_all = pd.concat([df[used_columns], lagged_fea, moving_avg], axis=1)
    return fea_all

def make_predictions(df, model):
    """Predict sales with the trained GBM model.
    
    Args: 
        df (Dataframe): Dataframe including all needed features
        model (Model): Trained GBM model
        
    Returns:
        Dataframe including the predicted sales of a certain store-brand
    """
    predictions = pd.DataFrame({'move': model.predict(df.drop('move', axis=1))})
    predictions['move'] = predictions['move'].apply(lambda x: round(x))
    return pd.concat([df[['brand', 'store', 'week']].reset_index(drop=True), predictions], axis=1)


if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, dest='data_folder', default='.', help='data folder mounting point')
    parser.add_argument('--num-leaves', type=int, dest='num_leaves', default=64, help='# of leaves of the tree')
    parser.add_argument('--min-data-in-leaf', type=int, dest='min_data_in_leaf', default=50, help='minimum # of samples in each leaf')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
    parser.add_argument('--feature-fraction', type=float, dest='feature_fraction', default=1.0, help='ratio of features used in each iteration')
    parser.add_argument('--bagging-fraction', type=float, dest='bagging_fraction', default=1.0, help='ratio of samples used in each iteration')
    parser.add_argument('--bagging-freq', type=int, dest='bagging_freq', default=1, help='bagging frequency')
    parser.add_argument('--max-rounds', type=int, dest='max_rounds', default=400, help='# of boosting iterations')
    parser.add_argument('--max-lag', type=int, dest='max_lag', default=10, help='max lag of unit sales')
    parser.add_argument('--window-size', type=int, dest='window_size', default=10, help='window size of moving average of unit sales')
    args = parser.parse_args()
    args.feature_fraction = round(args.feature_fraction, 2)
    args.bagging_fraction = round(args.bagging_fraction, 2)
    print(args)

    # Start an Azure ML run
    run = Run.get_context()

    # Data paths
    DATA_DIR = args.data_folder
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')

    # Data and forecast problem parameters
    TRAIN_START_WEEK = 40
    TRAIN_END_WEEK_LIST = list(range(135,159,2))
    TEST_START_WEEK_LIST = list(range(137,161,2))
    TEST_END_WEEK_LIST = list(range(138,162,2))
    # The start datetime of the first week in the record
    FIRST_WEEK_START = pd.to_datetime('1989-09-14 00:00:00')

    # Parameters of GBM model
    params = {
        'objective': 'mape', 
        'num_leaves': args.num_leaves, 
        'min_data_in_leaf': args.min_data_in_leaf, 
        'learning_rate': args.learning_rate, 
        'feature_fraction': args.feature_fraction, 
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq': args.bagging_freq,
        'num_rounds': args.max_rounds,
        'early_stopping_rounds': 125,
        'num_threads': 16
    }
    
    # Lags and used column names
    lags = np.arange(2, args.max_lag+1) 
    used_columns = ['store', 'brand', 'week', 'week_of_month', 'month', 'deal', 'feat', 'move', 'price', 'price_ratio']
    categ_fea = ['store', 'brand', 'deal']  

    # Train and validate the model using only the first round data
    r = 0
    print('---- Round ' + str(r+1) + ' ----')
    # Load training data
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_round_'+str(r+1)+'.csv'))
    train_df['move'] = train_df['logmove'].apply(lambda x: round(math.exp(x)))
    train_df = train_df[['store', 'brand', 'week', 'move']]

    # Create a dataframe to hold all necessary data
    store_list = train_df['store'].unique()
    brand_list = train_df['brand'].unique()
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
    data_filled['price_ratio'] = data_filled['price'] / data_filled['avg_price']
    data_filled.drop(price_cols, axis=1, inplace=True) 

    # Fill missing values
    data_filled = data_filled.groupby(['store', 'brand']).apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

    # Create datetime features
    data_filled['week_start'] = data_filled['week'].apply(lambda x: FIRST_WEEK_START + datetime.timedelta(days=(x-1)*7))
    data_filled['year'] = data_filled['week_start'].apply(lambda x: x.year)
    data_filled['month'] = data_filled['week_start'].apply(lambda x: x.month)
    data_filled['week_of_month'] = data_filled['week_start'].apply(lambda x: week_of_month(x))
    data_filled['day'] = data_filled['week_start'].apply(lambda x: x.day)
    data_filled.drop('week_start', axis=1, inplace=True)

    # Create other features (lagged features, moving averages, etc.)
    features = data_filled.groupby(['store','brand']).apply(lambda x: combine_features(x, ['move'], lags, args.window_size, used_columns))
    train_fea = features[features.week <= TRAIN_END_WEEK_LIST[r]].reset_index(drop=True)

    # Drop rows with NaN values
    train_fea.dropna(inplace=True)

    # Model training and validation 
    # Create a training/validation split
    train_fea, valid_fea, train_label, valid_label = train_test_split(train_fea.drop('move', axis=1, inplace=False), \
                                                                      train_fea['move'], test_size=0.05, random_state=1)
    dtrain = lgb.Dataset(train_fea, train_label)
    dvalid = lgb.Dataset(valid_fea, valid_label)
    # A dictionary to record training results 
    evals_result = {} 
    # Train GBM model
    bst = lgb.train(
        params, 
        dtrain, 
        valid_sets = [dtrain, dvalid], 
        categorical_feature = categ_fea, 
        evals_result = evals_result
    )
    # Get final training loss & validation loss
    train_loss = evals_result['training']['mape'][-1]
    valid_loss = evals_result['valid_1']['mape'][-1]
    print('Final training loss is {}'.format(train_loss))
    print('Final validation loss is {}'.format(valid_loss))

    # Log the validation loss/MAPE
    run.log('MAPE', np.float(valid_loss)*100)

    






