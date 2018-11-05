setwd('C://Users//vapaunic//repos//TSPerf')

args = commandArgs(trailingOnly=TRUE)
seed_value = args[1]
library('data.table')
library('gbm')

data_dir = 'energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/GBM/data/features'

train_dir = file.path(data_dir, 'train')
test_dir = file.path(data_dir, 'test')

train_file_prefix = 'train_round_'
test_file_prefix = 'test_round_'

output_file = file.path(paste('energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/GBM/submission_seed_', seed_value, '.csv', sep=""))

normalize_columns = list( 'LoadLag', 'DryBulbLag')

quantiles = seq(0.1, 0.9, by = 0.1)

result_all = list()
counter = 1

N_ROUNDS = 6
for (iR in 1:N_ROUNDS){
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
  
  ntrees = 1000
  shrinkage = 0.005

  for (z in zones) {
    print(paste('Zone', z))
    for (h in hours){
      train_df_sub = train_df[Zone == z & Hour == h]
      test_df_sub = test_df[Zone == z & Hour == h]
      
      result = data.table(Zone=test_df_sub$Zone, Datetime = test_df_sub$Datetime, Round=iR)
      
      for (tau in quantiles) {
        gbmModel = gbm(formula = DEMAND ~ LoadLag + DryBulbLag +
                       annual_sin_1 + annual_cos_1 + annual_sin_2 + annual_cos_2 + annual_sin_3 + annual_cos_3 +
                       weekly_sin_1 + weekly_cos_1 + weekly_sin_2 + weekly_cos_2 + weekly_sin_3 + weekly_cos_3,
                       distribution = list(name = "quantile", alpha = tau),
                       data = train_df_sub,
                       n.trees = ntrees,
                       shrinkage = shrinkage)

        gbmPredictions = predict(object = gbmModel,
                              newdata = test_df_sub,
                              n.trees = ntrees,
                              type = "response") * test_df_sub$LoadRatio

        result$Prediction = gbmPredictions
        result$q = tau
        
        result_all[[counter]] = result
        counter = counter + 1
      }
    }
  }
}

result_final = rbindlist(result_all)

fwrite(result_final, output_file)
