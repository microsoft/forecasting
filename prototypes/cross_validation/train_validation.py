import os, argparse
import pandas as pd
import numpy as np
from skgarden import RandomForestQuantileRegressor

from azureml.core import Run

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder',
                    help='data folder mounting point')
parser.add_argument('--min-samples-split', type=int,
                    dest='min_samples_split',
                    default=10)
parser.add_argument('--n-estimators', type=int, dest='n_estimators',
                    default=10)
args = parser.parse_args()

data_path = os.path.join(args.data_folder, 'train_round_1.csv')
min_samples_split = args.min_samples_split
n_estimators = args.n_estimators


def pinball_loss(predictions, actuals, q):
    zeros = pd.Series([0]*len(predictions))
    return (predictions-actuals).combine(zeros, max)*(1-q) + \
           (actuals-predictions).combine(zeros, max)*q

run = Run.get_submitted_run()

data_full = pd.read_csv(data_path, parse_dates=['Datetime'])

data_full = data_full.loc[data_full['Zone'] == 'CT']

train_validation_split = pd.to_datetime('2015-11-30 23:00:00')
train_data = data_full.loc[(data_full['Datetime'] <= train_validation_split)
                           & (data_full['Datetime'] >=
                              pd.to_datetime('2015-11-01 00:00:00'))]
validation_data = data_full.loc[data_full['Datetime'] > train_validation_split]

quantiles = np.linspace(0.1, 0.9, 9)
max_features = train_data.shape[1]//3
feature_cols = ['LoadLag', 'DryBulbLag',
                'annual_sin_1', 'annual_cos_1', 'annual_sin_2',
                'annual_cos_2', 'annual_sin_3', 'annual_cos_3',
                'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2',
                'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3'
                ]

rfqr = RandomForestQuantileRegressor(random_state=0,
                                     min_samples_split=min_samples_split,
                                     n_estimators=n_estimators,
                                     max_features=max_features)

rfqr.fit(train_data[feature_cols], train_data['DEMAND'])

result_list = []

for q in quantiles:
    q_predict = rfqr.predict(validation_data[feature_cols], quantile=q)

    result = pd.DataFrame({'predict': q_predict,
                           'q': q,
                           'actual': validation_data['DEMAND'],
                           'Datetime': validation_data['Datetime']})

    result_list.append(result)

result_final = pd.concat(result_list)

result_final.reset_index(inplace=True, drop=True)

result_final['loss'] = pinball_loss(result_final['predict'],
                                    result_final['actual'],
                                    result_final['q'])

average_pinball_loss = result_final['loss'].mean()

print('Average Pinball loss is {}'.format(average_pinball_loss))

run.log('average pinball loss', average_pinball_loss)
