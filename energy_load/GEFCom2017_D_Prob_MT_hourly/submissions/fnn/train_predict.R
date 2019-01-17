args = commandArgs(trailingOnly=TRUE)
# seed_value = args[1]
seed_value = 1
library('data.table')
library('qrnn')
library('doParallel')

cl <- parallel::makeCluster(4)
parallel::clusterEvalQ(cl, lapply(c("qrnn", "data.table"), library, character.only = TRUE))
registerDoParallel(cl)

data_dir = 'C:/Users/hlu/TSPerf/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/data/features'
train_dir = file.path(data_dir, 'train')
test_dir = file.path(data_dir, 'test')

train_file_prefix = 'train_round_'
test_file_prefix = 'test_round_'

output_file = file.path(paste('C:/Users/hlu/TSPerf/energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/fnn/submission_seed_', seed_value, '.csv', sep=""))

normalize_columns = list('LoadLag', 'DryBulbLag')

quantiles = seq(0.1, 0.9, by = 0.1)

result_all = list()
counter = 1

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
  

  zones = unique(train_df[, Zone])
  hours = unique(train_df[, Hour])
  
  test_df$AverageLoadRatio = rowMeans(test_df[,c('LoadRatio_10', 'LoadRatio_11', 'LoadRatio_12', 
                                                 'LoadRatio_13', 'LoadRatio_14', 'LoadRatio_15', 'LoadRatio_16')], na.rm=TRUE)
  
  
  test_df[, LoadRatio:=mean(AverageLoadRatio), by=list(Hour, MonthOfYear)]
  
  
  result_all_zones = foreach(z = zones, .combine = rbind) %dopar% {
    print(paste('Zone', z))
    
    result_all_hours = list()
    hour_counter = 1
    
    for (h in hours){
      train_df_sub = train_df[Zone == z & Hour == h]
      test_df_sub = test_df[Zone == z & Hour == h]
      
      result = data.table(Zone=test_df_sub$Zone, Datetime = test_df_sub$Datetime, Round=iR)
      
      train_x <- as.matrix(train_df_sub[, c('LoadLag', 'DryBulbLag',
                           'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 'annual_cos_2', 'annual_sin_3', 'annual_cos_3', 
                           'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3'), 
                           drop=FALSE])
      train_y <- as.matrix(train_df_sub[, c('DEMAND'), drop=FALSE])

      test_x <- as.matrix(test_df_sub[, c('LoadLag', 'DryBulbLag',
                           'annual_sin_1', 'annual_cos_1', 'annual_sin_2', 'annual_cos_2', 'annual_sin_3', 'annual_cos_3', 
                           'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2', 'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3'), 
                           drop=FALSE])
     
      result_all_quantiles = list()
      quantile_counter = 1

      for (tau in quantiles){

        model = qrnn2.fit(x=train_x, y=train_y, 
                          n.hidden=8, n.hidden2=4,
                          tau=tau, Th=tanh,
                          iter.max=1)
       
        result$Prediction = qrnn2.predict(model, x=test_x) * test_df_sub$LoadRatio
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

result_final = rbindlist(result_all)

fwrite(result_final, output_file)

parallel::stopCluster(cl)

