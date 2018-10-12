#!/usr/bin/Rscript 
#
# Seasonal Naive Method for Retail Forecasting Benchmark - OrangeJuice_Pt_3Weeks_Weekly
#
# This script can be executed with the following command
#       Rscript train_score.R -seed <seed value>
# where <seed value> is the random seed value from 1 to 5 (here since the forecast method
# is deterministic, this value will be simply used as a suffix of the output file name).

## Import packages used at the checkpoint time
library(optparse)
library(dplyr)
library(tidyr)
library(forecast)
library(MLmetrics)

## Define parameters
NUM_ROUNDS <- 12
TRAIN_START_WEEK <- 40
TRAIN_END_WEEK_LIST <- seq(135, 157, 2)
TEST_START_WEEK_LIST <- seq(137, 159, 2)
TEST_END_WEEK_LIST <- seq(138, 160, 2)
DATA_DIR <- './retail_sales/OrangeJuice_Pt_3Weeks_Weekly/data'

# Parse input argument
option_list <- list(
  make_option(c('-s', '--seed'), type='integer', default=NULL, 
              help='random seed value from 1 to 5', metavar='integer')
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

# Get the path of the current script and paths of data directories
SCRIPT_PATH <- file.path(dirname(DATA_DIR), 'submissions', 'SeasonalNaive')
TRAIN_DIR = file.path(DATA_DIR, 'train')
TEST_DIR = file.path(DATA_DIR, 'test')

# Generate submission file name
if (is.null(opt$seed)){
  output_file_name <- file.path(SCRIPT_PATH, 'submission.csv') 
  print('Random seed is not specified. Output file name will be submission.csv.')
} else{
  output_file_name <- file.path(SCRIPT_PATH, paste0('submission_seed_', as.character(opt$seed), '.csv'))
  print(paste0('Random seed is specified. Output file name will be submission_seed_', 
        as.character(opt$seed) , '.csv.'))
}

#### Implement snaive method on all the data  ####
pred_snaive_all <- list()
print('Using Seasonal Naive Method')

## snaive method 
apply_snaive_method <- function(train_sub) {
  cur_store <- train_sub$store[1]
  cur_brand <- train_sub$brand[1]
  train_ts <- ts(train_sub[c('logmove')], frequency = 52)
  pred_snaive <- snaive(train_ts, h=pred_horizon)
  pred_snaive_df <- data.frame(round = rep(r, pred_steps),
                                 store = rep(cur_store, pred_steps),
                                 brand = rep(cur_brand, pred_steps),
                                 week = pred_weeks,
                                 weeks_ahead = pred_weeks_ahead,
                                 prediction = round(exp(pred_snaive$mean[2:pred_horizon])))
}

for (r in 1:NUM_ROUNDS) { 
  print(paste0('---- Round ', r, ' ----'))
  pred_horizon <- TEST_END_WEEK_LIST[r] - TRAIN_END_WEEK_LIST[r]
  pred_steps <- TEST_END_WEEK_LIST[r] - TEST_START_WEEK_LIST[r] + 1
  pred_weeks <- TEST_START_WEEK_LIST[r]:TEST_END_WEEK_LIST[r]
  pred_weeks_ahead <- pred_weeks - TRAIN_END_WEEK_LIST[r]
  ## Import training data
  train_df <- read.csv(file.path(TRAIN_DIR, paste0('train_round_', as.character(r), '.csv')))
  ## Fill missing values
  store_list <- unique(train_df$store)
  brand_list <- unique(train_df$brand)
  week_list <- TRAIN_START_WEEK:TRAIN_END_WEEK_LIST[r]
  data_grid <- expand.grid(store = store_list,
                           brand = brand_list, 
                           week = week_list)
  train_filled <- merge(data_grid, train_df, 
                        by = c('store', 'brand', 'week'), 
                        all.x = TRUE)
  train_filled <- train_filled[,c('store','brand','week','logmove')]
  # Fill missing logmove 
  train_filled <- 
    train_filled %>% 
    group_by(store, brand) %>% 
    arrange(week) %>%
    fill(logmove) %>%
    fill(logmove, .direction = 'up')
  # Apply snaive method
  pred_snaive_all[[paste0('Round', r)]] <- 
    train_filled %>%
    group_by(store, brand) %>%
    do(apply_snaive_method(.))
}
# Combine and save forecast results
pred_snaive_all <- do.call(rbind, pred_snaive_all)
write.csv(pred_snaive_all, output_file_name, row.names = FALSE)

