#!/usr/bin/Rscript 
#
# This script trains Quantile Regression Neural Network models and evaluate the loss 
# on validation data of each cross validation round and forecast round with a set of 
# hyperparameters and calculate the average loss. 
# This script is used as the entry script for azureml hyperdrive.

args = commandArgs(trailingOnly=TRUE)

install.packages('qrnn', repo="http://cran.rstudio.com/")
install.packages('optparse', repo="http://cran.rstudio.com/")
library('data.table')
library('qrnn')
library("optparse")
library("rjson")
library('doParallel')

cl <- parallel::makeCluster(4)
parallel::clusterEvalQ(cl, lapply(c("qrnn", "data.table"), library, character.only = TRUE))
registerDoParallel(cl)

option_list = list(
  make_option(c("-d", "--path"), type="character", default=NULL,
              help="Path to the data files"),
  make_option(c("-c", "--cv_path"), type="character", default=NULL,
              help="Path to the cv setting files"),
  make_option(c("-n", "--n_hidden_1"), type="integer", default=NULL,
              help="Number of neurons in layer 1"),
  make_option(c("-m", "--n_hidden_2"), type="integer", default=NULL,
              help="Number of neurons in layer 2"),
  make_option(c("-i", "--iter_max"), type="integer", default=NULL,
              help="Number of maximum iterations"),
  make_option(c("-p", "--penalty"), type="integer", default=NULL,
              help="Penalty"),
  make_option(c("-t", "--time_stamp"), type="character", default=NULL,
              help="Timestamp")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser)

path = opt$path
cvpath = opt$cv_path
n.hidden = opt$n_hidden_1
n.hidden2= opt$n_hidden_2
iter.max = opt$iter_max
penalty = opt$penalty
ts = opt$time_stamp


# Data directory
train_dir = path
train_file_prefix = 'train_round_'


# Define cross validation split settings
cv_file = file.path(cvpath, 'cv_settings.json')
cv_settings = fromJSON(file=cv_file)


# Data and forecast parameters
normalize_columns = list('DEMAND_same_woy_lag', 'DryBulb_same_doy_lag')
quantiles = seq(0.1, 0.9, by = 0.1)


# Utility function
pinball_loss <- function(q, y, f) {
  L = ifelse(y>=f, q * (y-f), (1-q) * (f-y))
  return(L)
}


# Cross Validation
result_all = list()
counter = 1
for (i in 1:length(cv_settings)){
  round = paste("cv_round_", i, sep='')
  cv_settings_round = cv_settings[[round]]
  print(round)
  
  for (iR in 1:6){
    print(iR)
    
    train_file = file.path(train_dir, paste(train_file_prefix, as.character(iR), '.csv', sep=''))  
    cvdata_df = fread(train_file)
    
    cv_settings_cur = cv_settings_round[[as.character(iR)]]
    train_range = cv_settings_cur$train_range
    validation_range = cv_settings_cur$validation_range
    
    train_data = cvdata_df[Datetime >=train_range[1] & Datetime <= train_range[2]]
    validation_data = cvdata_df[Datetime >= validation_range[1] & Datetime <= validation_range[2]]
    
    zones = unique(validation_data$Zone)
    hours = unique(validation_data$Hour)
    
    for (c in normalize_columns){
      min_c = min(train_data[, ..c])
      max_c = max(train_data[, ..c])
      train_data[, c] = (train_data[, ..c] - min_c)/(max_c - min_c)
      validation_data[, c] = (validation_data[, ..c] - min_c)/(max_c - min_c)
    }
    
    validation_data$average_load_ratio = rowMeans(validation_data[, c('recent_load_ratio_10', 'recent_load_ratio_11', 'recent_load_ratio_12',
                                                                    'recent_load_ratio_13', 'recent_load_ratio_14', 'recent_load_ratio_15', 'recent_load_ratio_16')], na.rm=TRUE)
    validation_data[, load_ratio:=mean(average_load_ratio), by=list(Hour, month_of_year)]
    
    result_all_zones = foreach(z = zones, .combine = rbind) %dopar% {
      print(paste('Zone', z))
      
      features = c('DEMAND_same_woy_lag', 'DryBulb_same_doy_lag',
                   'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 
                   'annual_cos_2', 'annual_sin_3', 'annual_cos_3', 
                   'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 
                   'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3')
      subset_columns_train = c(features, 'DEMAND')
      subset_columns_validation = c(features, 'DEMAND', 'Zone', 'Datetime', 'load_ratio')
      
      result_all_hours = list()
      hour_counter = 1
      for (h in hours){
        train_df_sub = train_data[Zone == z & hour_of_day == h, ..subset_columns_train]
        validation_df_sub = validation_data[Zone == z & hour_of_day == h, ..subset_columns_validation]
        
        result = data.table(Zone=validation_df_sub$Zone, Datetime=validation_df_sub$Datetime, Round=iR, CVRound=i)
        
        train_x <- as.matrix(train_df_sub[, ..features, drop=FALSE])
        train_y <- as.matrix(train_df_sub[, c('DEMAND'), drop=FALSE])
        
        validation_x <- as.matrix(validation_df_sub[, ..features, drop=FALSE])
        
        result_all_quantiles = list()
        quantile_counter = 1
        
        for (tau in quantiles){
          
          model = qrnn2.fit(x=train_x, y=train_y, 
                            n.hidden=n.hidden, n.hidden2=n.hidden2,
                            tau=tau, Th=tanh,
                            iter.max=iter.max,
                            penalty=penalty)
          
          result$Prediction = qrnn2.predict(model, x=validation_x) * validation_df_sub$load_ratio
          result$DEMAND = validation_df_sub$DEMAND
          result$loss = pinball_loss(tau, validation_df_sub$DEMAND, result$Prediction)
          result$q = tau
          
          result_all_quantiles[[quantile_counter]] = result
          quantile_counter = quantile_counter + 1

        }
        result_all_hours[[hour_counter]] = rbindlist(result_all_quantiles)
        hour_counter = hour_counter + 1
      }
      rbindlist(result_all_hours)
    }
    result_all[[counter]] = result_all_zones
    counter = counter + 1
  }
}

result_final = rbindlist(result_all)

output_file_name = paste("cv_output_", ts, ".csv", sep = "")

output_file = file.path(paste(cvpath, '/',  output_file_name, sep=""))

fwrite(result_final, output_file)

parallel::stopCluster(cl)

