import os, argparse
import pandas as pd
import numpy as np
from skgarden import RandomForestQuantileRegressor
import json
from joblib import Parallel, delayed

from azureml.core import Run

NUM_CV_ROUND = 4
NUM_FORECAST_ROUND = 6

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

datetime_col = 'Datetime'
data_full = pd.read_csv(data_path, parse_dates=[datetime_col])

quantiles = np.linspace(0.1, 0.9, 9)
feature_cols = ['LoadLag', 'DryBulbLag',
                'annual_sin_1', 'annual_cos_1', 'annual_sin_2',
                'annual_cos_2', 'annual_sin_3', 'annual_cos_3',
                'weekly_sin_1', 'weekly_cos_1', 'weekly_sin_2',
                'weekly_cos_2', 'weekly_sin_3', 'weekly_cos_3'
                ]


def train_validate(train_data, validation_data):
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

    return average_pinball_loss


def train_single_group(train_df_single, group_name):
    model = RandomForestQuantileRegressor(random_state=0,
                                          min_samples_split=min_samples_split,
                                          n_estimators=n_estimators)

    model.fit(train_df_single[feature_cols], train_df_single['DEMAND'])

    return group_name, model


def predict_single_group(test_df_single, group_name, model):
    result_list = []

    for q in quantiles:
        q_predict = model.predict(test_df_single[feature_cols], quantile=q)

        result = pd.DataFrame({'predict': q_predict,
                               'q': q,
                               'actual': test_df_single['DEMAND'],
                               'Datetime': test_df_single['Datetime'],
                               'Zone': group_name})

        result_list.append(result)

    result_final = pd.concat(result_list)

    return result_final


def train(train_df, parallel):
    train_df_grouped = train_df.groupby('Zone')

    models_all = parallel\
        (delayed(train_single_group)(group.copy(), name)
         for name, group in train_df_grouped)

    models_all_dict = {}
    for k, v in models_all:
        models_all_dict[k] = v

    return models_all_dict


def predict(test_df, models_all, parallel):
    test_df_grouped = test_df.groupby('Zone')

    predictions = parallel\
        (delayed(predict_single_group)(group.copy(), name, models_all[name])
         for name, group in test_df_grouped)

    predictions_final = pd.concat(predictions)

    return predictions_final


with open('cv_settings_org.json') as f:
    cv_config = json.load(f)

predictions_all = []

with Parallel(n_jobs=-1) as parallel:
    for i_cv in range(1, NUM_CV_ROUND + 1):
        print('CV Round {}'.format(i_cv))
        cv_round = 'cv_round_' + str(i_cv)
        cv_config_round = cv_config[cv_round]

        for i_r in range(1, NUM_FORECAST_ROUND + 1):
            print('Forecast Round {}'.format(i_r))
            train_range = cv_config_round[str(i_r)]['train_range']
            validation_range = cv_config_round[str(i_r)]['validation_range']

            train_start = pd.to_datetime(train_range[0])
            train_end = pd.to_datetime(train_range[1])

            validation_start = pd.to_datetime(validation_range[0])
            validation_end = pd.to_datetime(validation_range[1])

            train_df = data_full.loc[(data_full[datetime_col] >= train_start)
                                     & (data_full[datetime_col] <= train_end)]
            validation_df = data_full.loc[(data_full[datetime_col] >=
                                           validation_start)
                                          & (data_full[datetime_col] <=
                                          validation_end)]

            validation_month = validation_df['MonthOfYear'].values[0]
            train_df = train_df.loc[train_df['MonthOfYear'] == validation_month,].copy()

            models_all = train(train_df, parallel)
            predictions_df = predict(validation_df, models_all, parallel)
            predictions_df['CVRound'] = i_cv
            predictions_df['ForecastRound'] = i_r
            predictions_all.append(predictions_df)

predictions_final = pd.concat(predictions_all)
predictions_final.reset_index(inplace=True, drop=True)

predictions_final['loss'] = pinball_loss(predictions_final['predict'],
                                         predictions_final['actual'],
                                         predictions_final['q'])

average_pinball_loss = predictions_final['loss'].mean()

print('Average Pinball loss is {}'.format(average_pinball_loss))

run.log('average pinball loss', average_pinball_loss)
