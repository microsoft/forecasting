"""
This .py file creates features for the RNN model.
"""

# import packages
import pandas as pd
import inspect, os
import numpy as np
import itertools
from utils import *
from sklearn.preprocessing import OneHotEncoder

import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs

data_relative_dir = '../../data'


def make_features(submission_round):
    # read in data
    file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    data_dir = os.path.join(file_dir, data_relative_dir)

    train_file = os.path.join(data_dir, 'train/train_round_{}.csv'.format(submission_round))
    test_file = os.path.join(data_dir, 'train/aux_round_{}.csv'.format(submission_round))

    train = pd.read_csv(train_file, index_col=False)
    test = pd.read_csv(test_file, index_col=False)

    # select the test data range for test data
    train_last_week = train['week'].max()
    test = test.loc[test['week'] > train_last_week]

    # calculate series popularity
    series_popularity = train.groupby(['store', 'brand']).apply(lambda x: x['logmove'].median())

    # fill the datetime gaps
    # such that every time series have the same length both in train and test
    store_list = train['store'].unique()
    brand_list = train['brand'].unique()
    train_week_list = range(bs.TRAIN_START_WEEK, bs.TRAIN_END_WEEK_LIST[submission_round - 1] + 1)
    test_week_list = range(bs.TEST_START_WEEK_LIST[submission_round - 1] - 1, bs.TEST_END_WEEK_LIST[submission_round - 1] + 1)

    train_item_list = list(itertools.product(store_list, brand_list, train_week_list))
    train_item_df = pd.DataFrame.from_records(train_item_list, columns=['store', 'brand', 'week'])

    test_item_list = list(itertools.product(store_list, brand_list, test_week_list))
    test_item_df = pd.DataFrame.from_records(test_item_list, columns=['store', 'brand', 'week'])

    train = train_item_df.merge(train, how='left', on=['store', 'brand', 'week'])
    test = test_item_df.merge(test, how='left', on=['store', 'brand', 'week'])

    # sort the train, test, series_popularity by store and brand
    train = train.sort_values(by=['store', 'brand', 'week'], ascending=True)
    test = test.sort_values(by=['store', 'brand', 'week'], ascending=True)
    series_popularity = series_popularity.reset_index().sort_values(
        by=['store', 'brand'], ascending=True).rename(columns={0: 'pop'})

    # calculate one-hot encoding for brands
    enc = OneHotEncoder(categories='auto')
    brand_train = np.reshape(train['brand'].values, (-1, 1))
    brand_test = np.reshape(test['brand'].values, (-1, 1))
    enc = enc.fit(brand_train)
    brand_enc_train = enc.transform(brand_train).todense()
    brand_enc_test = enc.transform(brand_test).todense()

    # calculate price and price_ratio
    price_cols = ['price1', 'price2', 'price3', 'price4', 'price5', 'price6', 'price7', 'price8',
                  'price9', 'price10', 'price11']

    train['price'] = train.apply(lambda x: x.loc['price' + str(int(x.loc['brand']))], axis=1)
    train['avg_price'] = train[price_cols].sum(axis=1).apply(lambda x: x / len(price_cols))
    train['price_ratio'] = train.apply(lambda x: x['price'] / x['avg_price'], axis=1)

    test['price'] = test.apply(lambda x: x.loc['price' + str(int(x.loc['brand']))], axis=1)
    test['avg_price'] = test[price_cols].sum(axis=1).apply(lambda x: x / len(price_cols))
    test['price_ratio'] = test.apply(lambda x: x['price'] / x['avg_price'], axis=1)

    # fill the missing values for feat, deal, price, price_ratio with 0
    for cl in ['price', 'price_ratio', 'feat', 'deal']:
        train.loc[train[cl].isna(), cl] = 0
        test.loc[test[cl].isna(), cl] = 0

    # normalize features:
    # 1) series popularity - 1
    # 2) brand - 11
    # 3) price: price and price_ratio
    # 4) promo: feat and deal
    series_popularity = series_popularity['pop'].values
    series_popularity = (series_popularity - series_popularity.mean()) \
                        / np.std(series_popularity)

    brand_enc_mean = brand_enc_train.mean(axis=0)
    brand_enc_std = brand_enc_train.std(axis=0)
    brand_enc_train = (brand_enc_train - brand_enc_mean) / brand_enc_std
    brand_enc_test = (brand_enc_test - brand_enc_mean) / brand_enc_std

    for cl in ['price', 'price_ratio', 'feat', 'deal']:
        cl_mean = train[cl].mean()
        cl_std = train[cl].std()
        train[cl] = (train[cl] - cl_mean) / cl_std
        test[cl] = (test[cl] - cl_mean) / cl_std

    # create the following numpy array
    # 1) ts_value_train (#ts, #train_ts_length)
    # 2) feature_train (#ts, #train_ts_length, #features)
    # 3) feature_test (#ts, #test_ts_length, #features)
    ts_number = len(series_popularity)

    train_min_time = bs.TRAIN_START_WEEK
    train_max_time = bs.TRAIN_END_WEEK_LIST[submission_round - 1]
    test_min_time = bs.TEST_START_WEEK_LIST[submission_round - 1] - 1
    test_max_time = bs.TEST_END_WEEK_LIST[submission_round - 1]
    train_ts_length = train_max_time - train_min_time + 1
    test_ts_length = test_max_time - test_min_time + 1

    # ts_value_train
    ts_value_train = train['logmove'].values
    ts_value_train = ts_value_train.reshape((ts_number, train_ts_length))
    # fill missing value with zero
    ts_value_train = np.nan_to_num(ts_value_train)

    # feature_train
    series_popularity_train = np.repeat(series_popularity, train_ts_length).reshape(
        (ts_number, train_ts_length, 1))

    brand_number = brand_enc_train.shape[1]
    brand_enc_train = np.array(brand_enc_train).reshape(
        (ts_number, train_ts_length, brand_number))
    price_promo_features_train = train[['price', 'price_ratio', 'feat', 'deal']].values.reshape(
        (ts_number, train_ts_length, 4))

    feature_train = np.concatenate((series_popularity_train, brand_enc_train, price_promo_features_train), axis=-1)

    # feature_test
    series_popularity_test = np.repeat(series_popularity, test_ts_length).reshape(
        (ts_number, test_ts_length, 1))

    brand_enc_test = np.array(brand_enc_test).reshape(
        (ts_number, test_ts_length, brand_number))
    price_promo_features_test = test[['price', 'price_ratio', 'feat', 'deal']].values.reshape(
        (ts_number, test_ts_length, 4))

    feature_test = np.concatenate((series_popularity_test, brand_enc_test, price_promo_features_test), axis=-1)

    # save the numpy arrays
    intermediate_data_dir = os.path.join(data_dir, 'intermediate/round_{}'.format(submission_round))
    if not os.path.isdir(intermediate_data_dir):
        os.makedirs(intermediate_data_dir)

    np.save(os.path.join(intermediate_data_dir, 'ts_value_train.npy'), ts_value_train)
    np.save(os.path.join(intermediate_data_dir, 'feature_train.npy'), feature_train)
    np.save(os.path.join(intermediate_data_dir, 'feature_test.npy'), feature_test)


