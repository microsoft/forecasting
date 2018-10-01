ptm <- proc.time()
library("optparse")
library("rjson")
library("data.table")
library("quantreg")

pinball_loss <- function(tau, y, q) {
  
  L = ifelse(y>=q, tau * (y-q), (1-tau) * (q-y))
  
  return(L)
}

option_list = list(
  make_option(c("-d", "--datapath"), type="character", default=NULL, 
              help="dataset file name", metavar="character"),
  make_option(c("-p", "--paramfile"), type="character", default=NULL, 
              help="name of the parameter setting file", metavar="character"),
  make_option(c("-c", "--cvfile"), type="character", default=NULL, 
              help="name of the cross validation setting file", metavar="character"),
  make_option(c("-s", "--paramset"), type="character", default=NULL, 
              help="parameter set to use", metavar="character")
); 

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser)

data_path = opt$datapath
parameter_file = opt$paramfile
cv_file = opt$cvfile
parameter_set = opt$paramset

print(data_path)
print(parameter_file)
print(cv_file)
print(parameter_set)

data_all = fread(data_path)
parameter_settings = fromJSON(file=parameter_file)
cv_settings = fromJSON(file=cv_file)

parameters = parameter_settings[[parameter_set]]
features = parameters$features
feature_set = parameters$feature_set
parameter_names = parameters$parameter_names
parameter_values = parameters$parameter_values

output_file_name = paste('feature_set_', feature_set, sep="")

for (i in 1:length(parameter_names)){
  output_file_name = paste(output_file_name, parameter_names[i], parameter_values[i], sep="_")
}

quantiles = seq(0.1, 0.9, by = 0.1)
normalize_columns = list( 'LoadLag', 'DryBulbLag')
subset_columns_train = c(features, 'DEMAND')
subset_columns_validation = c(features, 'DEMAND', 'Zone', 'Datetime', 'LoadRatio')

result_all = list()
counter = 1
for (i in 1:length(cv_settings)){
  round = paste("cv_round_", i, sep='')
  cv_settings_round = cv_settings[[round]]
  print(round)

  for (iR in 1:6){
    print(iR)
    cv_settings_cur = cv_settings_round[[as.character(iR)]]
    train_range = cv_settings_cur$train_range
    validation_range = cv_settings_cur$validation_range
    
    train_data = data_all[Datetime >=train_range[1] & Datetime <= train_range[2]]
    validation_data = data_all[Datetime >= validation_range[1] & Datetime <= validation_range[2]]
    
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
    
    for (z in zones) {
      print(paste('Zone', z))
      for (h in hours){
        train_df_sub = train_data[Zone == z & Hour == h, ..subset_columns_train]
        validation_df_sub = validation_data[Zone == z & Hour == h, ..subset_columns_validation]
        
        result = data.table(Zone=validation_df_sub$Zone, Datetime=validation_df_sub$Datetime, Round=iR, CVRound=i)
        
        for (tau in quantiles){
          
          model =  rq(DEMAND ~ ., data=train_df_sub, tau=tau, method=parameter_values[1])
          result$Prediction = predict(model, validation_df_sub) * validation_df_sub$LoadRatio
          result$DEMAND = validation_df_sub$DEMAND
          result$loss = pinball_loss(tau, validation_df_sub$DEMAND, result$Prediction)
          result$q = tau

          result_all[[counter]] = result
          counter = counter + 1
        }
      }
    }
  }
}

result_final = rbindlist(result_all)

average_PL = round(colMeans(result_final[, 'loss'], na.rm = TRUE), 2)
print(paste('Average Pinball Loss:', average_PL))

output_file_name = paste(output_file_name, 'APL', average_PL, sep="_")
output_file_name = paste(output_file_name, '.csv', sep="")

fwrite(result_final, output_file_name)

print(proc.time() - ptm)