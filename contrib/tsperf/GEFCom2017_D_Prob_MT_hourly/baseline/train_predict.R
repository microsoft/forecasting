args = commandArgs(trailingOnly=TRUE)
seed_value = args[1]
library('data.table')
library('quantreg')
library('doParallel')

n_cores = detectCores()

cl <- parallel::makeCluster(n_cores)
parallel::clusterEvalQ(cl, lapply(c("quantreg", "data.table"), library, character.only = TRUE))
registerDoParallel(cl)


data_dir = 'benchmarks/GEFCom2017_D_Prob_MT_hourly/baseline/data/features'
train_dir = file.path(data_dir, 'train')
test_dir = file.path(data_dir, 'test')

train_file_prefix = 'train_round_'
test_file_prefix = 'test_round_'

output_file = file.path(paste('benchmarks/GEFCom2017_D_Prob_MT_hourly/baseline/submission_seed_', seed_value, '.csv', sep=""))

normalize_columns = list('DEMAND_same_woy_lag', 'DryBulb_same_doy_lag')

quantiles = seq(0.1, 0.9, by = 0.1)

result_all = list()
for (iR in 1:6){
  print(paste('Round', iR))
  train_file = file.path(train_dir, paste(train_file_prefix, iR, '.csv', sep=''))
  test_file = file.path(test_dir, paste(test_file_prefix, iR, '.csv', sep=''))
  
  train_df = fread(train_file)
  test_df = fread(test_file)
  
  for (c in normalize_columns){
    min_c = min(train_df[, ..c])
    max_c = max(train_df[, ..c])
    train_df[, c] = (train_df[, ..c] - min_c)/(max_c - min_c)
    test_df[, c] = (test_df[, ..c] - min_c)/(max_c - min_c)
  }
  
  
  test_df$average_load_ratio = rowMeans(test_df[,c('recent_load_ratio_10', 'recent_load_ratio_11', 'recent_load_ratio_12',
                                                 'recent_load_ratio_13', 'recent_load_ratio_14', 'recent_load_ratio_15', 'recent_load_ratio_16')], na.rm=TRUE)
  test_df[, load_ratio:=mean(average_load_ratio), by=list(hour_of_day, month_of_year)]
  
  
  zones = unique(train_df[, Zone])
  hours = unique(train_df[, hour_of_day])
  all_zones_hours = expand.grid(zones, hours)
  colnames(all_zones_hours) = c('Zone', 'hour_of_day')
  
  result_all_zones_hours = foreach(i = 1:nrow(all_zones_hours), .combine = rbind) %dopar%{
    z = all_zones_hours[i, 'Zone']
    h = all_zones_hours[i, 'hour_of_day']
    
    train_df_sub = train_df[Zone == z & hour_of_day == h]
    test_df_sub = test_df[Zone == z & hour_of_day == h]
    
    result_all_quantiles = list()
    q_counter = 1
    for (tau in quantiles){
      result = data.table(Zone=test_df_sub$Zone, Datetime = test_df_sub$Datetime, Round=iR)
      
      model =  rq(DEMAND ~ DEMAND_same_woy_lag + DryBulb_same_doy_lag +
                    annual_sin_1 + annual_cos_1 + annual_sin_2 + annual_cos_2 + annual_sin_3 + annual_cos_3 +
                    weekly_sin_1 + weekly_cos_1 + weekly_sin_2 + weekly_cos_2 + weekly_sin_3 + weekly_cos_3,
                  data=train_df_sub, tau = tau)
      
      result$Prediction = predict(model, test_df_sub) * test_df_sub$load_ratio
      result$q = tau
      
      result_all_quantiles[[q_counter]] = result
      q_counter = q_counter + 1
      }
    rbindlist(result_all_quantiles)
  }
  result_all[[iR]] = result_all_zones_hours
}

result_final = rbindlist(result_all)
# Sort the quantiles
result_final = result_final[order(Prediction), q:=quantiles, by=c('Zone', 'Datetime', 'Round')]
result_final$Prediction = round(result_final$Prediction)

fwrite(result_final, output_file)
