"""
This .py file creates features for the RNN model.
"""

# import packages
import pandas as pd
import inspect, os
import numpy as np
import math
from sklearn.preprocessing import OneHotEncoder

# round number
ROUND = 1

# read in data
file_dir = './prototypes/retail_rnn_model'
# file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_relative_dir = '../../retail_sales/OrangeJuice_Pt_3Weeks_Weekly/data'
data_dir = os.path.join(file_dir, data_relative_dir)

train_file = os.path.join(data_dir, 'train/train_round_{}.csv'.format(ROUND))
test_file = os.path.join(data_dir, 'test/test_round_{}.csv'.format(ROUND))

train = pd.read_csv(train_file, index_col=False)
test = pd.read_csv(test_file, index_col=False)

# calculate move - weekly sales
train['move'] = train['logmove'].apply(lambda x: math.exp(x))

# mark the logmove and profit in the test dataset to prevent data leakage issue
test['logmove'] = np.nan
test['profit'] = np.nan

# calculate series popularity
series_popularity = train.groupby(['store', 'brand']).apply(lambda x: x['logmove'].median())


# fill the datetime gaps
def fill_datetime_gap(df, min_week=None):
    if not min_week:
        min_week = df['week'].min()
    week_list = list(range(min_week, df['week'].max() + 1))
    week_list_df = pd.DataFrame({'week': week_list})
    df = week_list_df.merge(df, how='left', on='week')
    new_col = [cl for cl in df.columns if cl not in ['store', 'brand']]
    return df[new_col]


train_filled = train.groupby(['store', 'brand']).apply(lambda df: fill_datetime_gap(df))
train_filled = train_filled.reset_index(level=[0, 1]).reset_index(drop=True)
train_week_max = train['week'].max()
test_filled = test.groupby(['store', 'brand']).apply(lambda df: fill_datetime_gap(df, train_week_max + 1))
test_filled = test_filled.reset_index(level=[0, 1]).reset_index(drop=True)

# calculate one-hot encoding for brands
pd.get_dummies(train_filled['brand'].astype(dtype='int32'), prefix='brand')
enc = OneHotEncoder()

train_brand = np.reshape(train_filled['brand'].values, (-1, 1))
test_brand = np.reshape(test_filled['brand'].values, (-1, 1))
enc = enc.fit(train_brand)
train_brand_enc = enc.transform(train_brand)
test_brand_enc = enc.transform(test_brand)

# fill the missing values for feat, deal, price, price_ratio with 0

