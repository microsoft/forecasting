#!/usr/bin/Rscript 
#
# Select the best ARIMA model for Retail Forecasting Benchmark - OrangeJuice_Pt_3Weeks_Weekly
#
# This script can be executed with the following command from TSPerf directory
#                         Rscript <submission dir>/model_selection.r 
# It outputs a csv file containing the orders of the best ARIMA models selected by auto.arima 
# function in R package forecast. 

# Import packages
library(dplyr)
library(tidyr)
library(forecast)

# Define parameters
NUM_ROUNDS <- 12
TRAIN_START_WEEK <- 40
TRAIN_END_WEEK_LIST <- seq(135, 157, 2)

# Paths of the training data and submission folder
DATA_DIR <- './retail_sales/OrangeJuice_Pt_3Weeks_Weekly/data'
TRAIN_DIR <- file.path(DATA_DIR, 'train')
SUBMISSION_DIR <- file.path(dirname(DATA_DIR), 'submissions', 'ARIMA')

#### Select ARIMA model for every store-brand based on 1st-round training data ####
print('Selecting ARIMA models')
arima_model_all <- list()

select_arima_model <- function(train_sub) {
  # Selects the best ARIMA model for the time series of each store-brand. 
  # 
  # Args:
  #   train_sub (Dataframe): Training data of a certain store-brand
  # 
  # Returns:
  #   arima_order_df (Dataframe): Configuration of the best ARIMA model
  cur_store <- train_sub$store[1]
  cur_brand <- train_sub$brand[1]
  train_ts <- ts(train_sub[c('logmove')], frequency = 52)
  fit_arima <- auto.arima(train_ts)
  arima_order <- arimaorder(fit_arima)
  arima_order_df <- data.frame(store = cur_store,
                               brand = cur_brand,
                               seasonal = length(arima_order) > 3,
                               p = arima_order['p'],
                               d = arima_order['d'],
                               q = arima_order['q'],
                               P = arima_order['P'],
                               D = arima_order['D'],
                               Q = arima_order['Q'],
                               m = arima_order['Frequency'])
}

r = 1
print(paste0('---- Round ', r, ' ----'))
# Import training data
train_df <- read.csv(file.path(TRAIN_DIR, paste0('train_round_', as.character(r), '.csv')))

# Create a dataframe to hold all necessary data
store_list <- unique(train_df$store)
brand_list <- unique(train_df$brand)
week_list <- TRAIN_START_WEEK:TRAIN_END_WEEK_LIST[r]
data_grid <- expand.grid(store = store_list,
                          brand = brand_list, 
                          week = week_list)
train_filled <- merge(data_grid, train_df, 
                      by = c('store', 'brand', 'week'), 
                      all.x = TRUE)
train_filled <- train_filled[, c('store','brand','week','logmove')]

# Fill missing logmove 
train_filled <- 
  train_filled %>% 
  group_by(store, brand) %>% 
  arrange(week) %>%
  fill(logmove) %>%
  fill(logmove, .direction = 'up')

# Select ARIMA models
arima_model_all <- 
  train_filled %>%
  group_by(store, brand) %>%
  do(select_arima_model(.))

# Combine and save model selection results
write.csv(arima_model_all, file.path(SUBMISSION_DIR, 'hparams.csv'), row.names = FALSE)

