import os
from os.path import join
import argparse
import pandas as pd
from numpy import arange
from ensemble_parallel import RandomForestQuantileRegressor
import time

# get seed value 
parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
parser.add_argument('--seed', type=int, dest='seed', help='random seed')
args = parser.parse_args()

# initialize location of input and output files
data_dir = join(args.data_folder, 'features')
train_dir = join(data_dir, 'train')
test_dir = join(data_dir, 'test')
output_file = join('outputs','submission_seed_{}.csv'.format(args.seed))

# do 6 rounds of forecasting, at each round output 9 quantiles
n_rounds = 6   
quantiles = arange(0.1,1,0.1)

# schema of the output
y_test = pd.DataFrame(columns=['Datetime','Zone','Round','q','Prediction'])

for i in range(1,n_rounds+1):
    print('Round {}'.format(i))

    # read training and test files for the current round
    train_file = join(train_dir, 'train_round_{}.csv'.format(i))
    train_df = pd.read_csv(train_file)

    test_file = join(test_dir, 'test_round_{}.csv'.format(i))
    test_df = pd.read_csv(test_file)

    # train and test for each hour separately
    for hour in arange(0,24):
        print(hour)

        # select training sets
        train_df_hour = train_df[(train_df['Hour']==hour)]
        train_df_hour = pd.get_dummies(train_df_hour, columns=['Zone'])   # create one-hot encoding of Zone (scikit-garden works only with numerical columns)
        X_train = train_df_hour.drop(columns=['Datetime','DEMAND','DryBulb','DewPnt']).values    # remove column that are not useful (Datetime) or are not 
                                                                                                 # available in the test set (DEMAND, DryBulb, DewPnt)
        y_train = train_df_hour['DEMAND'].values

        # train a model
        rfqr = RandomForestQuantileRegressor(random_state=args.seed, n_jobs=-1,
                                             n_estimators=1000, max_features='sqrt', max_depth=12)
        rfqr.fit(X_train, y_train)

        # select test set
        test_df_hour = test_df[test_df['Hour']==hour]
        y_test_baseline = test_df_hour[['Datetime','Zone']]
        test_df_cat = pd.get_dummies(test_df_hour, columns=['Zone'])
        X_test = test_df_cat.drop(columns=['Datetime']).values

        # generate forecast for each quantile
        percentiles = rfqr.predict(X_test,quantiles*100)
        for j, quantile in enumerate(quantiles):
            y_test_round_quantile = y_test_baseline.copy(deep=True)
            y_test_round_quantile['Round'] = i
            y_test_round_quantile['q'] = quantile
            y_test_round_quantile['Prediction'] = percentiles[:,j]
            y_test = pd.concat([y_test, y_test_round_quantile])

# store forecasts
os.makedirs('outputs', exist_ok=True)
y_test.to_csv(output_file,index=False)
