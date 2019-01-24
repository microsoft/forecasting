# coding: utf-8

# Create input features for the boosted decision tree model.  

import os
import sys
import math
import itertools
import datetime
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb 

# Append TSPerf path to sys.path
tsperf_dir = os.getcwd()
if tsperf_dir not in sys.path:
    sys.path.append(tsperf_dir)

# Import TSPerf components
from utils import *
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

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
    
    Returns:
        fea (Dataframe): Dataframe consisting of the moving averages
    """
    if window_size == None: # Use a large window to compute average over all historical data
        window_size = df.shape[0]
    fea = df.shift(start_step).rolling(min_periods=1, center=False, window=window_size).mean()
    fea.columns = fea.columns + '_mean'
    return fea

def combine_features(df, lag_fea, lags, max_moving_step, used_columns):
    """Combine different features for a certain store and brand.
    
    Args:
        df (Dataframe): Time series data of a certain store and brand
        lag_fea (List): A list of column names for creating lagged features
        lags (Numpy Array): Numpy array including all the lags
        max_moving_step (Integer): Maximum step for computing the moving average
        used_columns (List): A list of names of columns used in model training (including target variable)
    
    Returns:
        fea_all (Dataframe): Dataframe including all features for the specific store and brand
    """
    lagged_fea = lagged_features(df[lag_fea], lags)
    moving_avg = moving_averages(df[lag_fea], 2, max_moving_step)
    fea_all = pd.concat([df[used_columns], lagged_fea, moving_avg], axis=1)
    return fea_all


def make_features(pred_round, train_dir, lags, max_moving_step, used_columns, store_list, brand_list):
    """Create a dataframe of the input features.
    
    Args: 
        pred_round (Integer): Prediction round
        train_dir (String): Path of the training data directory 
        lags (Numpy Array): Numpy array including all the lags
        max_moving_step (Integer): Maximum step for computing the moving average
        used_columns (List): A list of names of columns used in model training (including target variable)
        store_list (Numpy Array): List of all the store IDs 
        brand_list (Numpy Array): List of all the brand IDs 
        
    Returns:
        features (Dataframe): Dataframe including all the input features and target variable 
    """ 
    # Load training data
    train_df = pd.read_csv(os.path.join(train_dir, 'train_round_'+str(pred_round+1)+'.csv'))
    train_df['move'] = train_df['logmove'].apply(lambda x: round(math.exp(x)))
    train_df = train_df[['store', 'brand', 'week', 'profit', 'move']]

    # Create a dataframe to hold all necessary data
    week_list = range(bs.TRAIN_START_WEEK, bs.TEST_END_WEEK_LIST[pred_round]+1)
    d = {'store': store_list,
        'brand': brand_list,
        'week': week_list}        
    data_grid = df_from_cartesian_product(d)
    data_filled = pd.merge(data_grid, train_df, how='left', 
                            on=['store', 'brand', 'week'])

    # Get future price, deal, and advertisement info
    aux_df = pd.read_csv(os.path.join(train_dir, 'aux_round_'+str(pred_round+1)+'.csv'))  
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
    data_filled['week_start'] = data_filled['week'].apply(lambda x: bs.FIRST_WEEK_START + datetime.timedelta(days=(x-1)*7))
    data_filled['year'] = data_filled['week_start'].apply(lambda x: x.year)
    data_filled['month'] = data_filled['week_start'].apply(lambda x: x.month)
    data_filled['week_of_month'] = data_filled['week_start'].apply(lambda x: week_of_month(x))
    data_filled['day'] = data_filled['week_start'].apply(lambda x: x.day)
    data_filled.drop('week_start', axis=1, inplace=True)

    # Create other features (lagged features, moving averages, etc.)
    features = data_filled.groupby(['store','brand']).apply(lambda x: combine_features(x, ['move'], lags, max_moving_step, used_columns))

    return features