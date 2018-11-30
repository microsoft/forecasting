# Problem

Sales forecasting is a key task for the management of retail stores. With the projection of future sales, store managers will be able to optimize 
the inventory based on their business goals. This will generate more profitable order fulfillment and reduce the inventory cost. 

The task of this benchmark is to forecast orange juice sales of different brands for multiple stores with the Orange Juice (OJ) dataset from R package 
`bayesm`. The forecast type is point forecasting. The forecast horizon is 3 weeks ahead and granularity is weekly. There are 12 forecast rounds, each of 
which involves forecasting the sales during a target period. The training and test data in each round are specified in the subsection [Training and test data 
separation](#training-and-test-data-separation). The table below summarizes the characteristics of this benchmark

|  |  |
| ----------------------------------- | - |
| **Number of time series**           | 913 |
| **Forecast frequency**   | every two weeks |
| **Forecast granularity**         | weekly |
| **Forecast type**                   | point |

# Data

## Dataset attribution

The OJ dataset is from R package [bayesm](https://cran.r-project.org/web/packages/bayesm/index.html) and is part of the [Dominick's dataset](https://www.chicagobooth.edu/research/kilts/datasets/dominicks). 

## Dataset description

This dataset contains the following two tables:

1. Weekly sales of refrigerated orange juice at 83 stores. This table has 106139 rows and 19 columns. It includes weekly sales and prices of 11 orange juice 
brands as well as information about profit, deal, and advertisement for each brand. Note that the weekly sales is captured by a column named `logmove` which 
corresponds to the natural logarithm of the number of units sold. To get the number of units sold, you need to apply an exponential transform to this column.

2. Demographic information on those stores. This table has 83 rows and 13 columns. For every store, the table describes demographic information of its consumers, 
distance to the nearest warehouse store, average distance to the nearest 5 supermarkets, ratio of its sales to the nearest warehouse store, and ratio of its sales 
to the average of the nearest 5 stores.

Note that the week number starts from 40 in this dataset, while the full Dominick's dataset has week number from 1 to 400. According to [Dominick's Data Manual](https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/dominicks-manual-and-codebook_kiltscenter.aspx), week 1 starts on 09/14/1989.
Please see pages 40 and 41 of the [bayesm reference manual](https://cran.r-project.org/web/packages/bayesm/bayesm.pdf) and the [Dominick's Data Manual](https://www.chicagobooth.edu/-/media/enterprise/centers/kilts/datasets/dominicks-dataset/dominicks-manual-and-codebook_kiltscenter.aspx) for more details about the data. 
 

## Training and test data separation

For this benchmark, you are provided successive folds of training data in 12 forecast rounds. The goal is to generate forecasts for the forecast periods listed 
in the table below, using the available training data:

| **Round** | **Train period start week** | **Train period end week** | **Forecast period start week** | **Forecast period end week** |
| -------- | --------------- | ------------------ | ------------------------- | ----------------------- |
| 1 | 40 | 135 | 137 | 138 |
| 2 | 40 | 137 | 139 | 140 |
| 3 | 40 | 139 | 141 | 142 |
| 4 | 40 | 141 | 143 | 144 |
| 5 | 40 | 143 | 145 | 146 |
| 6 | 40 | 145 | 147 | 148 |
| 7 | 40 | 147 | 149 | 150 |
| 8 | 40 | 149 | 151 | 152 |
| 9 | 40 | 151 | 153 | 154 |
| 10 | 40 | 153 | 155 | 156 |
| 11 | 40 | 155 | 157 | 158 |
| 12 | 40 | 157 | 159 | 160 |

The gap of one week between training period and forecasting period allows the store managers to prepare the 
stock to meet the forecasted demand. Besides, we assume that the information about the price, deal, and advertisement up until the forecast period end week is available in each round. 

# Format of Forecasts

The forecasts should be in the following format

| round | store | brand | week | weeks_ahead | prediction | 
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| ... | ... | ... | ... | ... | ... |

with each of the columns explained below 
* round: index of the forecast round
* store: store number
* brand: brand indicator
* week: week of the sales that we forecast
* weeks_ahead: number of weeks ahead that we forecast
* prediction: predicted number of units sold  
 


# Quality

**Evaluation metric**: Mean Absolute Percentage Error (MAPE)  
**Minimum performance**: TBD  

# Instructions to run reference implementation
**TODO: Fill this section when we have a draft reference implementation**

# Instructions to submit new benchmark implementations
**TODO: Fill this section when we have a draft reference implementation**

