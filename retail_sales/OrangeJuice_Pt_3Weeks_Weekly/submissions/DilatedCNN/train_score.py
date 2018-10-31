#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, subprocess

def get_gpu_name():
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)

def get_cuda_version():
    """Get CUDA version"""
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
        # This breaks on linux:
        #cuda=!ls "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        #path = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\" + str(cuda[0]) +"\\version.txt"
    elif sys.platform == 'linux' or sys.platform == 'darwin':
        path = '/usr/local/cuda/version.txt'
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data = f.read().replace('\n','')
        return data
    else:
        return "No CUDA in this machine"
    
print(get_gpu_name())
print(get_cuda_version())


# In[2]:


import keras


# In[3]:


import os
import sys
import time
import math
import keras
import datetime
import itertools
import numpy as np
import pandas as pd

from keras.layers import * 
from keras.models import Model
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler


# In[4]:


# Append TSPerf path to sys.path
nb_dir = os.path.split(os.getcwd())[0]
tsperf_dir = os.path.dirname(os.path.dirname(os.path.dirname(nb_dir)))
if tsperf_dir not in sys.path:
    sys.path.append(tsperf_dir)

from common.metrics import MAPE
import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs


# In[5]:


# Data paths
DATA_DIR = '../../data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Data parameters
MAX_STORE_ID = 137
MAX_BRAND_ID = 11

# Parameters of the model
PRED_HORIZON = 3
PRED_STEPS = 2
SEQ_LEN = 8
DYNAMIC_FEATURES = ['deal', 'feat']
STATIC_FEATURES = ['store', 'brand']


# In[6]:


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

def gen_sequence(df, seq_len, seq_cols, start_timestep=0, end_timestep=None):
    """Reshape features into an array of dimension (time steps, features).  
    
    Args:
        df (Dataframe): Time series data of a specific (store, brand) combination
        seq_len (Integer): The number of previous time series values to use as input features
        seq_cols (List): A list of names of the feature columns 
        start_timestep (Integer): First time step you can use to create feature sequences
        end_timestep (Integer): Last time step you can use to create feature sequences
        
    Returns:
        A generator object for iterating all the feature sequences
    """
    data_array = df[seq_cols].values
    if end_timestep is None:
        end_timestep = df.shape[0]
    for start, stop in zip(range(start_timestep, end_timestep-seq_len+1), range(start_timestep+seq_len, end_timestep+1)):
        yield data_array[start:stop, :]


def gen_sequence_array(df_all, seq_len, seq_cols, start_timestep=0, end_timestep=None):
    """Combine feature sequences for all the combinations of (store, brand) into an 3d array.
    
    Args:
        df_all (Dataframe): Time series data of all stores and brands
        seq_len (Integer): The number of previous time series values to use as input features
        seq_cols (List): A list of names of the feature columns 
        start_timestep (Integer): First time step you can use to create feature sequences
        end_timestep (Integer): Last time step you can use to create feature sequences
        
    Returns:
        seq_array (Numpy Array): An array of the feature sequences of all stores and brands    
    """
    seq_gen = (list(gen_sequence(df_all[(df_all['store']==cur_store) & (df_all['brand']==cur_brand)],                                  seq_len, seq_cols, start_timestep, end_timestep))               for cur_store, cur_brand in itertools.product(df_all['store'].unique(), df_all['brand'].unique()))
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    return seq_array

def static_feature_array(df_all, total_timesteps, seq_cols):
    """Generate an arary which encodes all the static features.
    """
    fea_df = data_filled.groupby(['store', 'brand']).                          apply(lambda x: x.iloc[:total_timesteps,:]).                          reset_index(drop=True)
    fea_array = fea_df[seq_cols].values
    return fea_array


# In[7]:


r = 0
print('---- Round ' + str(r+1) + ' ----')
train_df = pd.read_csv(os.path.join(TRAIN_DIR, 'train_round_'+str(r+1)+'.csv'))
train_df['move'] = train_df['logmove'].apply(lambda x: round(math.exp(x)))
train_df.drop('logmove', axis=1, inplace=True)
print(train_df.head(3))
print('')
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
print('Number of missing rows is {}'.format(data_filled[data_filled.isnull().any(axis=1)].shape[0]))
print('')
data_filled = data_filled.groupby(['store', 'brand']).                           apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
print(data_filled.head(3))
print('')


# In[8]:


# Normalize the dataframe of features
cols_normalize = data_filled.columns.difference(['store','brand','week'])
min_max_scaler = MinMaxScaler()
data_filled_scaled = pd.DataFrame(min_max_scaler.fit_transform(data_filled[cols_normalize]), 
                                  columns=cols_normalize, 
                                  index=data_filled.index)
data_filled_scaled.head()


# In[ ]:





# In[9]:


data_sub = data_filled[(data_filled.brand==1) & (data_filled.store==2)]
data_sub.shape


# In[10]:


start_timestep = 0
end_timestep = bs.TRAIN_END_WEEK_LIST[r]-bs.TRAIN_START_WEEK-PRED_HORIZON+1

train_input1 = gen_sequence_array(data_filled, SEQ_LEN, ['move'], start_timestep, end_timestep)

train_input1.shape


# In[11]:


start_timestep = PRED_HORIZON
end_timestep = bs.TRAIN_END_WEEK_LIST[r]-bs.TRAIN_START_WEEK+1

train_input2 = gen_sequence_array(data_filled, SEQ_LEN, DYNAMIC_FEATURES, start_timestep, end_timestep)

train_input2.shape


# In[12]:


train_input = np.concatenate((train_input1, train_input2), axis=2)
train_input.shape


# In[13]:


#train_output
start_timestep = SEQ_LEN+1
end_timestep = bs.TRAIN_END_WEEK_LIST[r]-bs.TRAIN_START_WEEK+1
train_output = gen_sequence_array(data_filled, PRED_STEPS, ['move'], start_timestep, end_timestep)
train_output = np.squeeze(train_output)
train_output.shape


# In[ ]:





# In[14]:


store_col = np.full([train_input1.shape[0],1], 1)
brand_col = np.full([train_input1.shape[0],1], 2)
np.concatenate((store_col, brand_col), axis=1)


# In[15]:


# Test MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(data_sub)
data_sub_scaled = scaler.transform(data_sub)
data_sub_scaled


# In[16]:


data_sub_scaled[0,]


# In[17]:


total_timesteps = bs.TRAIN_END_WEEK_LIST[r]-bs.TRAIN_START_WEEK-SEQ_LEN-PRED_HORIZON+2
train_input2 = static_feature_array(data_filled, total_timesteps, ['store', 'brand'])


# In[18]:


train_input2.shape


# In[19]:


# Model definition
def create_dcnn_model(seq_len, kernel_size=2, n_filters=4, n_input_series=1, n_outputs=1):
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
    conv_out = Dropout(0.25)(conv_out)
    conv_out = Flatten()(conv_out)
    
    # Concatenate with categorical features
    x = concatenate([conv_out, Flatten()(store_embed), Flatten()(brand_embed)])
    #x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(n_outputs, activation='linear')(x)
    
    model = Model(inputs=[seq_in, cat_fea_in], outputs=output)
    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='mse', optimizer=adam, metrics=['mae'])
    return model


# In[20]:


model = create_dcnn_model(seq_len=SEQ_LEN, n_outputs=PRED_STEPS)
model.summary()


# In[ ]:


model.fit([train_input1, train_input2], train_output, epochs=2, batch_size=2, validation_data=([train_input1, train_input2], train_output))


# In[ ]:


np.mean(train_output)


# In[ ]:


from keras.utils import multi_gpu_model

try:
    model = multi_gpu_model(model)
    print("Training using multiple GPUs..")
except:
    print("Training using single GPU or CPU..")

start_time = time.time()
adam = optimizers.Adam(lr=0.01)
model.compile(loss='mse', optimizer=adam, metrics=['mae'])
model.fit([train_input1, train_input2], train_output, epochs=5, batch_size=2, validation_data=([train_input1, train_input2], train_output))
print(time.time()-start_time)


# In[ ]:


model = multi_gpu_model(model)


# In[ ]:




