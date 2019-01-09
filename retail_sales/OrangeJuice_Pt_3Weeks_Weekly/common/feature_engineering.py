import os
import math
import itertools
import pandas as pd
import datetime
from benchmark_paths import DATA_DIR
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs


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
    wom = int(ceil(adjusted_dom / 7.0))
    return wom


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
    if window_size == None:  # Use a large window to compute average over all historical data
        window_size = df.shape[0]
    fea = df.shift(start_step).rolling(min_periods=1, center=False, window=window_size).mean()
    fea.columns = fea.columns + '_mean'
    return fea


if __name__ == '__main__':
    # define the round
    submission_round = 1

    # read in data
    train_file = os.path.join(DATA_DIR, 'train/train_round_{}.csv'.format(submission_round))
    aux_file = os.path.join(DATA_DIR, 'train/aux_round_{}.csv'.format(submission_round))
    train_df = pd.read_csv(train_file, index_col=False)
    aux_df = pd.read_csv(aux_file, index_col=False)

    # calculate move
    train_df['move'] = train_df['logmove'].apply(lambda x: round(math.exp(x)))
    train_df = train_df[['store', 'brand', 'week', 'profit', 'move', 'logmove']]

    # merge train_df with aux_df
    all_df = pd.merge(train_df, aux_df, how='right', on=['store', 'brand', 'week'])

    # fill missing datetime gaps
    store_list = all_df['store'].unique()
    brand_list = all_df['brand'].unique()
    week_list = range(bs.TRAIN_START_WEEK, bs.TEST_END_WEEK_LIST[submission_round - 1] + 1)

    item_list = list(itertools.product(store_list, brand_list, week_list))
    item_df = pd.DataFrame.from_records(item_list, columns=['store', 'brand', 'week'])
    all_df = item_df.merge(all_df, how='left', on=['store', 'brand', 'week'])

    # calculate features
    # (1) price and price ratio
    # Create relative price feature
    price_cols = ['price1', 'price2', 'price3', 'price4', 'price5', 'price6', 'price7', 'price8', \
                  'price9', 'price10', 'price11']
    all_df['price'] = all_df.apply(lambda x: x.loc['price' + str(int(x.loc['brand']))], axis=1)
    all_df['avg_price'] = all_df[price_cols].sum(axis=1).apply(lambda x: x / len(price_cols))
    all_df['price_ratio'] = all_df['price'] / all_df['avg_price']

    # (2) week of month
    all_df['week_start'] = all_df['week'].apply(
        lambda x: bs.FIRST_WEEK_START + datetime.timedelta(days=(x - 1) * 7))
    all_df['week_of_month'] = all_df['week_start'].apply(lambda x: week_of_month(x))

    # (3) lag features and moving average features
    lags = [2, 3, 4]
    lagged_fea = lagged_features(all_df[['move']], lags)
    moving_avg = moving_averages(all_df[['move']], 2, 10)
    tmp = pd.concat([all_df, lagged_fea, moving_avg], axis=1)

    # feature columns
    feature_cols = ['store', 'brand', 'week', 'profit', 'move', 'logmove', 'price1',
       'price2', 'price3', 'price4', 'price5', 'price6', 'price7', 'price8',
       'price9', 'price10', 'price11', 'deal', 'feat', 'price', 'avg_price',
       'price_ratio', 'week_start', 'week_of_month', 'move_lag2', 'move_lag3',
       'move_lag4', 'move_mean']

    # write the feature engineering results to csv
    



