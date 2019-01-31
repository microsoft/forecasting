#!/usr/bin/Rscript 
#
# This script trains Quantile Regression Neural Network models and evaluate the loss 
# on validation data of each cross validation round and forecast round with a set of 
# hyperparameters and calculate the average loss. 
# This script is used for grid search on vm.

args = commandArgs(trailingOnly=TRUE)
parameter_set = args[1]

install.packages('rjson', repo="http://cran.r-project.org/")
install.packages('doParallel', repo="http://crloads the featured training and test data in, normalizes the features, 
# partitions the data by zone and hour, andan.r-project.org/")
library('data.table')
library('qrnn')
library('rjson')
library('doParallel')

cl <- parallel::makeCluster(4)
parallel::clusterEvalQ(cl, lapply(c("qrnn", "data.table"), library, character.only = TRUE))
registerDoParallel(cl)

# Specify data directory
data_dir = 'energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/data/features'
train_dir = file.path(data_dir, 'train')

train_file_prefix = 'train_round_'

# Define parameter grid
n.hidden_choice = c(4, 8)
n.hidden2_choice = c(4, 8)
iter.max_choice = c(1, 2, 4, 6, 8)
penalty_choice = c(0, 0.001)

param_grid = expand.grid(n.hidden_choice,
                         n.hidden2_choice,
                         iter.max_choice,
                         penalty_choice)

colnames(param_grid) = c('n.hidden', 'n.hidden2', 'iter.max', 'penalty')
parameter_names = colnames(param_grid)
parameter_values = param_grid[parameter_set, ]

output_file_name = 'cv_output'
for (j in 1:length(parameter_names)){
  output_file_name = paste(output_file_name, parameter_names[j], parameter_values[j], sep="_")
}

output_file = file.path(paste('energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/', output_file_name, sep=""))

# Define cross validation split settings
cv_file = file.path(paste('energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/', 'cv_settings.json', sep=""))
cv_settings = fromJSON(file=cv_file)

# Parameters of model
n.hidden = as.integer(param_grid[parameter_set, 'n.hidden'])
n.hidden2 = as.integer(param_grid[parameter_set, 'n.hidden2'])
iter.max = as.integer(param_grid[parameter_set, 'iter.max'])
penalty = as.integer(param_grid[parameter_set, 'penalty'])

# Data and forecast parameters
features = c('LoadLag', 'DryBulbLag',
             'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 
             'annual_cos_2', 'annual_sin_3', 'annual_cos_3', 
             'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 
             'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3')

normalize_columns = list('LoadLag', 'DryBulbLag')
quantiles = seq(0.1, 0.9, by = 0.1)
subset_columns_train = c(features, 'DEMAND')
subset_columns_validation = c(features, 'DEMAND', 'Zone', 'Datetime', 'LoadRatio')

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
    
    validation_data$AverageLoadRatio = rowMeans(validation_data[,c('LoadRatio_10', 'LoadRatio_11', 'LoadRatio_12', 
                                                                   'LoadRatio_13', 'LoadRatio_14', 'LoadRatio_15', 'LoadRatio_16')], na.rm=TRUE)
    validation_data[, LoadRatio:=mean(AverageLoadRatio), by=list(Hour, MonthOfYear)]
    
    result_all_zones = foreach(z = zones, .combine = rbind) %dopar% {
      print(paste('Zone', z))

      result_all_hours = list()
      hour_counter = 1
      
      for (h in hours){
        train_df_sub = train_data[Zone == z & Hour == h, ..subset_columns_train]
        validation_df_sub = validation_data[Zone == z & Hour == h, ..subset_columns_validation]
        
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
       
          result$Prediction = qrnn2.predict(model, x=validation_x) * validation_df_sub$LoadRatio
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

average_PL = round(colMeans(result_final[, 'loss'], na.rm = TRUE), 2)
print(paste('Average Pinball Loss:', average_PL))

output_file_name = paste(output_file_name, 'APL', average_PL, sep="_")
output_file_name = paste(output_file_name, '.csv', sep="")

output_file = file.path(paste('energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/', output_file_name, sep=""))

fwrite(result_final, output_file)

parallel::stopCluster(cl)
