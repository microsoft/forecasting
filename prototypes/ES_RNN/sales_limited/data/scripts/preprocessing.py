#script to transform retail sales data to a format similar to the data provided for the 
#M4 competition. This data will be fed naively into slawek's winning ES-RNN model.

import pandas as pd
import numpy as np
import math

#parse file name from command line
training_files = './training-files.txt'

#open file and get list of all files to read/transform
f = open(training_files)
# info = str(f.getline().strip())
num_files = int(f.readline().strip())
filelist = []
for i in range(num_files): 
    filelist.append(f.readline().strip())

print(filelist)
for filename in filelist: 
    #extract round number
    round_num = ''
    numset = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}
    for i in range(len(filename) - 1, -1, -1): #assumes that round number is the only number occurring in filename
        if filename[i] in numset: 
            round_num = filename[i] + round_num
        if filename[i] == '/': #done traversing true name of current file
            break
    round_num = int(round_num)
    print('Starting round #' + str(round_num))

    #open file
    yx = pd.read_csv(filename)

    #ascertain what range of weeks the file covers
    min_week = min(set(yx.iloc[:,2]))
    max_week = max(set(yx.iloc[:,2]))
    
    #create a new dataframe to store the processed data
    col_list = ['SalesID', 'Brand', 'store and brand'] + ['week ' + str(x) for x in range(min_week, max_week + 1)]
    df = pd.DataFrame([], columns=col_list)

    store_list = sorted(list(set(yx.iloc[:, 0])))
    brand_list = sorted(list(set(yx.iloc[:, 1])))

    rows_needed = len(store_list) * len(brand_list)
    
    for i in range(rows_needed): 
        df2 = pd.Series([float('nan')] * (3 + max_week - min_week + 1), index=col_list)
        df = df.append(df2, ignore_index=True)

    count = 0
    tsMap = dict() #map from (store, brand) to index in df
    for i in store_list: 
        for j in brand_list: 
            tsMap[(i, j)] = count
            count += 1

    print('creating dataframe...')
    #iterate through rows of current file and populate new file
    for i, val in yx.iterrows(): 
        store_brand_pair = (val['store'], val['brand'])
        df_index = tsMap[store_brand_pair]
        df['Brand'][df_index] = val['brand']
        df['store and brand'][df_index] = store_brand_pair
        week_col = 'week ' + str(int(val['week']))
        df[week_col][df_index] = math.exp(val[3]) #math.exp(logmove) = number of units sold

    for i, val in df.iterrows(): 
        df['SalesID'][i] = 'E' + str(i+1)

    print('dataframe successfully created')
    #separate between time-series and information
    dftimeseries = df[['SalesID'] + ['week ' + str(x) for x in range(min_week, max_week + 1)]]
    dfinfo = df[['SalesID', 'Brand', 'store and brand']]

    print('replacing nan values')
        #forward and backward fill data in timeseries dataframe
        #data fill algo
    for i, val in dftimeseries.iterrows():
        #begin by attempting to forward fill (i.e. use past values to fill in future NaNs)
        for wk in range(min_week, max_week + 1): 
            current_week = wk
            while pd.isnull(dftimeseries['week ' + str(wk)][i]): 
                if current_week > min_week: 
                    current_week -= 1
                    dftimeseries['week ' + str(wk)][int(i)] = dftimeseries['week ' + str(current_week)][int(i)]
                else: 
                    break
    print('forward fill done!')
        #back fill (i.e. use future values to fill in previous NaNs)
    for i, val in dftimeseries.iterrows():
        for wk in range(max_week, min_week - 1, -1): 
            current_week = wk
            while pd.isnull(dftimeseries['week ' + str(wk)][i]): 
                if current_week < max_week: 
                    current_week += 1
                    dftimeseries['week ' + str(wk)][int(i)] = dftimeseries['week ' + str(current_week)][int(i)]
                else: 
                    break
    
    #save csvs
    print('saving csv files...')
    dftimeseries.to_csv('./../Train/ts_train_round_' + str(round_num) + '.csv', index=False)
    # dfinfo.to_csv('sales-info.csv', index=False) #we need only create the info csv for the whole dataset
    print('Done with round ' + str(round_num) + '!----------------------------------------')

f.close()