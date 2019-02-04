#script to transform the output of the naive (unmodified) ES-RNN model to the form specified 
#in the sales_data readme.md file.

import numpy as np
import pandas as pd

#helper function adjust headings for outputted forecasts
def transformer(outframe): 
    #add current headers as last column
    last_entry = list(outframe.keys())
      #make a new data frame for last col and append it to input df
    append_frame = pd.DataFrame([last_entry], columns=last_entry)
    updated = outframe.append(append_frame, ignore_index=True)
    
    #rename column headings
      #build map of old column headings to their replacement
    rename_map = dict()
    keys = list(outframe.keys())
    for i in range(len(keys)): 
        rename_map[keys[i]] = 'V' + str(i+1)
    rename_map
    
      #modify headers to mirror that of the dataframe
    updated = updated.rename(index=int, columns=rename_map)
    
    return updated

#helper function to format transform output into form desired by forecast evaluator
def truth_creator(df): 
    #create new df with appropriate columns
    col_list = ['round', 'store', 'brand', 'week', 'weeks ahead', 'prediction']
    truth = pd.DataFrame(columns=['round', 'store', 'brand', 'week', 'weeks ahead', 'prediction'])
    
    for i, val in df.iterrows(): 
        #get code
        entry_code = val['V1']
        
        #row in info df
        info_row_index = int(entry_code[1:]) - 1
        info_row = info.iloc[info_row_index]
        
        #extract brand and store from info dataframe
        brand = info_row['Brand']
        store_and_brand = info_row['store and brand'][1:]
        store = ''
        for i in store_and_brand:
            if i != ',': 
                store += i
            else: 
                break
        store = int(float(store))
        
        #get week number and number of weeks ahead
        forecasts = ['V2', 'V3', 'V4']
        for x in range(len(forecasts)):
            weeks_ahead = x + 1
            week = 160 + weeks_ahead
            prediction = val[forecasts[x]]
            temp = pd.Series([float('nan'), store, brand, week, weeks_ahead, prediction], index=col_list)
            truth = truth.append(temp, ignore_index=True)
        
    return truth

#main
#parse file name from command line
test_files = './test_files.txt'

#open file and get list of all files to read/transform
f = open(test_files)
info = str(f.getline().strip())
num_files = int(f.readline().strip())
filelist = []
for i in range(num_files): 
    filelist.append(f.readline())


for filename in filelist: 
    test = pd.read_csv(filename)

    test = transformer(test)
    truth = truth_creator(test)
    truth.to_csv(filename[0:len(filename) - 4] + '_modified.csv')
