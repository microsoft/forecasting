"""
This .py file creates features for the RNN model.
"""

# import packages
import pandas as pd
import inspect, os
import math

# read in data
file_dir = './prototypes/retail_rnn_model'
# file_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_relative_dir = '../../retail_sales/OrangeJuice_Pt_3Weeks_Weekly\data'
data_dir = os.path.join(file_dir, data_relative_dir)
sales_file = os.path.join(data_dir, 'yx.csv')
sales = pd.read_csv(sales_file, index_col=False)

# calculate move - weekly sales
sales['move'] = sales['logmove'].apply(lambda x: math.exp(x))

# calculate series popularity
series_popularity = sales.groupby(['store', 'brand']).apply(lambda x: x['logmove'].median())


# fill the datetime gaps
def fill_datetime_gap(df):
    week_list = list(range(df['week'].min(), df['week'].max() + 1))
    week_list_df = pd.DataFrame({'week': week_list})
    df = week_list_df.merge(df, how='left', on='week')
    return df


# fill the missing values for feat, deal, price, price_ratio with 0

# calculate one-hot encoding for brands
