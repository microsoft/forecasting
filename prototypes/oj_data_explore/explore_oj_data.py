
# coding: utf-8

# In[37]:


# import packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# In[4]:


# read in the data
data_dir = '../../retail_sales/OrangeJuice_Pt_3Weeks_Weekly\data'
sales_file = os.path.join(data_dir, 'yx.csv')
store_file = os.path.join(data_dir, 'storedemo.csv')
# Read sales data into dataframe
sales = pd.read_csv(sales_file, index_col=False)
store = pd.read_csv(store_file, index_col=False)


# In[5]:


# merge the dataset
sales = sales.merge(store, how='left', left_on='store', right_on='STORE')


# In[6]:


sales.columns


# In[7]:


# how many time series and how long
print('number of stores...')
print(len(sales.groupby(['store']).groups.keys()))
print('number of brands...')
print(len(sales.groupby(['brand']).groups.keys()))
print('number of time series...')
print(len(sales.groupby(['store', 'brand']).groups.keys()))
print('lenth distribution of the time series...')
sales.groupby(['store', 'brand']).size().describe()


# In[9]:


# investigate into profit
import math
sales['move'] = sales['logmove'].apply(lambda x: math.exp(x))
tmp = sales[['store', 'brand', 'move', 'profit', 'price1', 'deal', 'feat']].loc[sales['brand'] == 1]
tmp['profit/move'] = tmp['profit'] / tmp['move'] 


# In[27]:


# initial conclusion: the correlation between the sales vs these store features seems week.
# store related features vs sales
for cl in store.columns[1:]:
    #plt.scatter(sales[cl], sales['logmove'])
    p = sales.boxplot(column='logmove', by=cl)
    p.axes.get_xaxis().set_visible(False)


# In[33]:


# sales across different brands and stores
for by_cl in ['brand', 'store', ['brand', 'store']]:
    p = sales.boxplot(column='logmove', by=by_cl)
    p.axes.get_xaxis().set_visible(False)


# In[38]:


# autocorrealtion: weekly, monthly, quarterly, yearly
def single_autocorr(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """
    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2
    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0


# In[57]:


store_list = sales['store'].unique()
brand_list = sales['brand'].unique()
l_range = range(1, 53)

for i in range(len(store_list)):
    for j in range(len(brand_list)):
        store = store_list[i]
        brand = brand_list[j]
        d = sales.loc[(sales['store'] == store) & (sales['brand'] == brand)]
        cor = []
        for l in l_range:
            cor.append(single_autocorr(d['logmove'], l))
        plt.scatter(l_range, cor)
        plt.show()


# In[47]:


for l in [13, 26, 39, 52]:
    print(l)
    print(sales.groupby(['store', 'brand']).apply(lambda x: single_autocorr(x['logmove'], l)).describe())


# In[63]:


# correlation between deal, feat
plt.scatter(sales['feat'], sales['logmove'])
p = sales.boxplot(column='logmove', by='deal')


# In[93]:


# correlation between the competitive prices
sales['price'] = sales.apply(lambda x: x.loc['price' + str(int(x.loc['brand']))], axis=1)


# In[102]:


sales[list(sales.columns[:16]) + ['price']]


# In[104]:


sales.columns

