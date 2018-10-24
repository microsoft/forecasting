
# coding: utf-8

# Train and score a boosted decision tree model using [LightGBM Python package](https://github.com/Microsoft/LightGBM) from Microsoft, 
# which is a fast, distributed, high performance gradient boosting framework based on decision tree algorithms.  


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
#cur_dir = os.path.split(os.getcwd())[0]
#tsperf_dir = os.path.dirname(os.path.dirname(os.path.dirname(cur_dir)))
tsperf_dir = os.getcwd()
if tsperf_dir not in sys.path:
    sys.path.append(tsperf_dir)

from common.metrics import MAPE
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

# Get random seed
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, help='Random seed of GRM algorithm')
args = parser.parse_args()
seed = args.seed
print('Random seed is {}'.format(seed))

# Data paths
DATA_DIR = os.path.join(tsperf_dir, 'retail_sales', 'OrangeJuice_Pt_3Weeks_Weekly', 'data')
SUBMISSION_DIR = os.path.join(tsperf_dir, 'retail_sales', 'OrangeJuice_Pt_3Weeks_Weekly', 'submissions', 'LightGBM')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Parameters of GBM model
params = {
    'num_leaves': 50, 
    'objective': 'mape', 
    'min_data_in_leaf': 200, 
    'learning_rate': 0.002, 
    'feature_fraction': 0.9, 
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'num_threads': 16,
    'seed': seed
}
MAX_ROUNDS = 100 

# Lags and categorical features
lags = [2,3,4] 
lags_str = [str(x) for x in lags]
categ_fea = ['store', 'brand', 'deal', 'feat'] 

first_week_start = pd.to_datetime('1989-09-07 00:00:00') 

# Utility functions
def week_of_month(dt):
    """ 
    Get the week of the month for the specified date.
    
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
    """
    Compute averages of every feature over moving time windows.
    
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

def create_features(df):
    """
    Create features used for model training.
    
    Args:
        df (Dataframe): Time series data of a certain store and brand
    
    Returns:
        fea_all (Dataframe): All features for the specific store and brand
    """
    lagged_fea = lagged_features(df[['move']], lags)
    moving_avg = moving_averages(df[['move']], 2, 10)
    fea_columns = ['brand' , 'store', 'week', 'week_of_month', 'day', 'profit', 'deal' , 'feat', 'move']
    #fea_columns = fea_columns + ['price1', 'price2', 'price3', 'price4', 'price5', 'price6', 'price7', 'price8', 'price9', 'price10', 'price11']
    fea_all = pd.concat([df[fea_columns], lagged_fea, moving_avg], axis=1)
    return fea_all

def make_predictions(df, model):
    """
    Predict sales with the trained GBM model.
    
    Args: 
        df (Dataframe): Dataframe including all needed features
        model (Model): Trained GBM model
        
    Returns:
        Dataframe including the predicted sales of a certain store and brand
    """
    predictions = pd.DataFrame({'move': model.predict(df.drop('move', axis=1))})
    predictions['move'] = predictions['move'].apply(lambda x: round(x))
    return pd.concat([df[['brand', 'store', 'week']].reset_index(drop=True), predictions], axis=1)

def evaluate(result):
    """
    Compute MAPE value of the forecast.
    
    Args:
        result (Dataframe): Input dataframe including predicted sales and actual sales
    
    Returns:
        MAPE value of the forecast
    """
    return MAPE(result['move'], result['actual'])*100

# Train and predict for all forecast rounds
pred_all = []
metric_all = []
for r in range(bs.NUM_ROUNDS): 
    print('---- Round ' + str(r+1) + ' ----')
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_round_'+str(r+1)+'.csv'))
    train_df['move'] = train_df['logmove'].apply(lambda x: round(math.exp(x)))
    train_df.drop('logmove', axis=1, inplace=True)
    #print(train_df.head(3))
    #print('')
    # Fill missing values
    store_list = train_df['store'].unique()
    brand_list = train_df['brand'].unique()
    week_list = range(bs.TRAIN_START_WEEK, bs.TEST_END_WEEK_LIST[r]+1)
    d = {'store': store_list,
         'brand': brand_list,
         'week': week_list}        
    data_grid = df_from_cartesian_product(d)
    data_filled = pd.merge(data_grid, train_df, how='left', 
                            on=['store', 'brand', 'week'])
    #print('Number of missing rows is {}'.format(data_filled[data_filled.isnull().any(axis=1)].shape[0]))
    #print('')
    data_filled = data_filled.groupby(['store', 'brand']).apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    # Create datetime features
    data_filled['week_start'] = data_filled['week'].apply(lambda x: first_week_start + datetime.timedelta(days=(x-bs.TRAIN_START_WEEK)*7))
    data_filled['year'] = data_filled['week_start'].apply(lambda x: x.year)
    data_filled['month'] = data_filled['week_start'].apply(lambda x: x.month)
    data_filled['week_of_month'] = data_filled['week_start'].apply(lambda x: week_of_month(x))
    data_filled['day'] = data_filled['week_start'].apply(lambda x: x.day)
    data_filled.drop('week_start', axis=1, inplace=True)
    # Create other features (lagged features, moving averages, etc.)
    features = data_filled.groupby(['store','brand']).apply(lambda x: create_features(x))
    train_fea = features[features.week <= bs.TRAIN_END_WEEK_LIST[r]].reset_index(drop=True)
    # Drop rows with NaN values
    train_fea.dropna(inplace=True)
    #print(train_fea.head(1))
    #print('')
    print('Training and predicting models...')
    evals_result = {} # to record eval results for plotting
    dtrain = lgb.Dataset(
                train_fea.drop('move', axis=1, inplace=False), 
                label = train_fea['move']
    )
    # Train GBM model
    bst = lgb.train(
        params, 
        dtrain, 
        num_boost_round = MAX_ROUNDS,
        valid_sets = [dtrain], 
        categorical_feature = categ_fea,
        early_stopping_rounds = 125, 
        evals_result = evals_result,
        verbose_eval = False
    )
    # Generate forecasts
    test_fea = features[features.week >= bs.TEST_START_WEEK_LIST[r]].reset_index(drop=True)
    pred = make_predictions(test_fea, bst).sort_values(by=['store','brand', 'week']).reset_index(drop=True)
    # Additional columns required by the submission format
    pred['round'] = r+1
    pred['weeks_ahead'] = pred['week'] - bs.TRAIN_END_WEEK_LIST[r]
    #print(pred)
    #print('')
    ## Evaluate prediction accuracy
    #test_df = pd.read_csv(os.path.join(TEST_DIR, 'test_round_'+str(r+1)+'.csv'))
    #test_df['actual'] = test_df['logmove'].apply(lambda x: round(math.exp(x)))
    #test_df.drop('logmove', axis=1, inplace=True)
    #combined = pd.merge(pred, test_df, on=['store', 'brand', 'week'], how='left')
    #metric_value = evaluate(combined)
    #print('')
    #print('MAPE of current round is {}'.format(metric_value))
    #print('')
    # Keep the predictions and accuracy
    pred_all.append(pred)
    #metric_all.append(metric_value)

# Generate submission
submission = pd.concat(pred_all, axis=0)
submission.rename(columns={'move': 'prediction'}, inplace=True)
submission = submission[['round', 'store', 'brand', 'week', 'weeks_ahead', 'prediction']]
filename = 'submission_seed_' + str(seed) + '.csv'
submission.to_csv(os.path.join(SUBMISSION_DIR, filename), index=False)
#print('MAPE of the submission is {}'.format(np.mean(metric_all))) 



