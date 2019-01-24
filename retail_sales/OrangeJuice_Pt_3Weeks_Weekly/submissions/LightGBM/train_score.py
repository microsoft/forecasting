
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
tsperf_dir = os.getcwd()
if tsperf_dir not in sys.path:
    sys.path.append(tsperf_dir)

from make_features import make_features
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

def make_predictions(df, model):
    """Predict sales with the trained GBM model.
    
    Args: 
        df (Dataframe): Dataframe including all needed features
        model (Model): Trained GBM model
        
    Returns:
        Dataframe including the predicted sales of a certain store and brand
    """
    predictions = pd.DataFrame({'move': model.predict(df.drop('move', axis=1))})
    predictions['move'] = predictions['move'].apply(lambda x: round(x))
    return pd.concat([df[['brand', 'store', 'week']].reset_index(drop=True), predictions], axis=1)

if __name__ == '__main__':
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, dest='seed', default=1, help='Random seed of GBM model')
    parser.add_argument('--num-leaves', type=int, dest='num_leaves', default=32, help='# of leaves of the tree')
    parser.add_argument('--min-data-in-leaf', type=int, dest='min_data_in_leaf', default=50, help='mini # of samples in each leaf')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
    parser.add_argument('--feature-fraction', type=float, dest='feature_fraction', default=1.0, help='ratio of features used in each iteration')
    parser.add_argument('--bagging-fraction', type=float, dest='bagging_fraction', default=1.0, help='ratio of samples used in each iteration')
    parser.add_argument('--bagging-freq', type=int, dest='bagging_freq', default=1, help='bagging frequency')
    parser.add_argument('--max-rounds', type=int, dest='max_rounds', default=400, help='# of boosting iterations')
    parser.add_argument('--max-lag', type=int, dest='max_lag', default=5, help='max lag of unit sales')
    parser.add_argument('--max-moving-step', type=int, dest='max_moving_step', default=5, help='max step for computing moving average of unit sales')
    args = parser.parse_args()
    print(args)

    # Data paths
    DATA_DIR = os.path.join(tsperf_dir, 'retail_sales', 'OrangeJuice_Pt_3Weeks_Weekly', 'data')
    SUBMISSION_DIR = os.path.join(tsperf_dir, 'retail_sales', 'OrangeJuice_Pt_3Weeks_Weekly', 'submissions', 'LightGBM')
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')

    # Parameters of GBM model
    params = {
        'num_leaves': 113, #50, 
        'objective': 'mape', 
        'min_data_in_leaf': 500, #200, 
        'learning_rate': 0.1, #0.002, 
        'feature_fraction': 0.28, #0.9, 
        'bagging_fraction': 0.7,
        'bagging_freq': 18, #1,
        'num_threads': 16,
        'early_stopping_rounds': 125,
        'seed': args.seed
    }
    MAX_ROUNDS = 1720 #400 

    # Lags and categorical features
    lags = np.arange(2, 19+1)
    used_columns = ['store', 'brand', 'week', 'week_of_month', 'month', 'deal', 'feat', 'move', 'price', 'price_ratio']
    categ_fea = ['store', 'brand', 'deal'] 

    # Get unique stores and brands
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_round_1.csv'))
    store_list = train_df['store'].unique()
    brand_list = train_df['brand'].unique()

    # Train and predict for all forecast rounds
    pred_all = []
    metric_all = []
    for r in range(bs.NUM_ROUNDS): 
        print('---- Round ' + str(r+1) + ' ----')
        # Create features
        features = make_features(r, TRAIN_DIR, lags, args.max_moving_step, used_columns, store_list, brand_list)

        train_fea = features[features.week <= bs.TRAIN_END_WEEK_LIST[r]].reset_index(drop=True)

        # Drop rows with NaN values
        train_fea.dropna(inplace=True)

        print('Training models and making predictions...')
        dtrain = lgb.Dataset(train_fea.drop('move', axis=1, inplace=False), 
                            label = train_fea['move'])
        # Train GBM model
        bst = lgb.train(
            params, 
            dtrain, 
            num_boost_round = MAX_ROUNDS,
            valid_sets = [dtrain], 
            categorical_feature = categ_fea,
            verbose_eval =  True #False
        )
        # Generate forecasts
        test_fea = features[features.week >= bs.TEST_START_WEEK_LIST[r]].reset_index(drop=True)
        pred = make_predictions(test_fea, bst).sort_values(by=['store','brand', 'week']).reset_index(drop=True)
        # Additional columns required by the submission format
        pred['round'] = r+1
        pred['weeks_ahead'] = pred['week'] - bs.TRAIN_END_WEEK_LIST[r]
        # Keep the predictions 
        pred_all.append(pred)

    # Generate submission
    submission = pd.concat(pred_all, axis=0)
    submission.rename(columns={'move': 'prediction'}, inplace=True)
    submission = submission[['round', 'store', 'brand', 'week', 'weeks_ahead', 'prediction']]
    filename = 'submission_seed_' + str(seed) + '.csv'
    submission.to_csv(os.path.join(SUBMISSION_DIR, filename), index=False)



