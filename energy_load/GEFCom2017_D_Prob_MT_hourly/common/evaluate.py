"""
This script evaluates an implementation of the 
GEFCom2017-D_Prob_MT_hourly benchmark. It reads in
the test set predictions contained in submission.csv
file that should accompany every reference implementation
and submission. It computes evaluation metrics by
comparing the predictions to the actual true values
contained within the test set.

The script must be executed from the TSPerf root directory.

Arguments:
    submission_file:   relative file path to submission.csv 
        to the reference or submission implementation
"""


import os
import sys
import pandas as pd
sys.path.append('.')
from benchmark_paths import BENCHMARK_DIR
from common.loss_functions import pinball_loss

# baseline losses are taken from Table 2 of
# F. Ziel. Quantile regression for the qualifying match of GEFCom2017 probabilistic load forecasting. International Journal of Forecasting, 2018
zones = ['CT','MA_TOTAL','ME','NEMA','NH','RI','SEMA','TOTAL','VT','WCMA']
baseline_losses = pd.DataFrame({'1':[114.88, 170.20, 36.95, 77.85, 41.91, 23.32, 44.11, 402.68, 22.44, 50.58],\
                                '2':[115.72, 190.36, 29.11, 81.02, 35.34, 24.18, 50.69, 401.51, 15.49, 60.32],\
                                '3':[115.72, 190.36, 29.11, 81.02, 35.34, 24.18, 50.69, 401.51, 15.49, 60.32],\
                                '4':[98.91, 175.86, 23.96, 73.32, 29.43, 21.54, 49.62, 351.89, 21.07, 55.43],\
                                '5':[98.80, 175.86, 23.88, 73.16, 29.64, 21.53, 49.51, 351.70, 20.92, 55.25],\
                                '6':[55.11, 106.5, 29.71, 44.41, 16.74, 11.19, 34.19, 202.83, 17.23, 34.91]},\
                                index=zones)

def read_test_files(benchmark_dir):
    
    test_data_dir = os.path.join(benchmark_dir, "data", "test_ground_truth")
    for rnd in range(1, 7):
        test_file = 'test_round_'+str(rnd)+'.csv'
        test_round = pd.read_csv(os.path.join(test_data_dir, test_file), parse_dates=['Datetime'])
        test_round['Round'] = rnd
        test_round = test_round[['Round', 'Datetime', 'Zone', 'DEMAND']]
        if rnd > 1:
            test = test.append(test_round)
        else:
            test = test_round.copy()
    return test

def evaluate(submission_file):
    test = read_test_files(BENCHMARK_DIR)

    print(os.path.join(BENCHMARK_DIR, submission_file))
    submission = pd.read_csv(os.path.join(BENCHMARK_DIR, submission_file), parse_dates=['Datetime'])

    evaluation = pd.merge(submission, test, on=['Round', 'Datetime', 'Zone'], how='left')
    evaluation['pinball'] = pinball_loss(evaluation['Prediction'], evaluation['DEMAND'], evaluation['q'])
    print("Mean pinball loss: ", evaluation['pinball'].mean())

    submission_losses = evaluation[['Zone', 'pinball']].groupby('Zone').mean().rename(columns={'pinball': 'mean pinball loss'})
    print(submission_losses)

    # for each round, compute relative improvement over baseline
    n_rounds = 6
    rel_improvements = pd.DataFrame(index=zones)
    rel_improvements.index.name = 'Zone'
    for r in range(1,n_rounds+1):
        evaluation_round = evaluation[evaluation['Round']==r]
        submission_losses = evaluation_round[['Zone', 'pinball']].groupby('Zone').mean()
        rel_improvement_round = (baseline_losses[str(r)]-submission_losses['pinball'])/baseline_losses[str(r)]*100
        rel_improvements['Round '+str(r)] = rel_improvement_round.values

    print("\nRelative improvement (in %) over GEFCom2017 benchmark model")
    print(rel_improvements)
    print("\nAverage relative improvement (in %) over GEFCom2017 benchmark model")
    print(rel_improvements.mean().to_string())

if __name__ == "__main__":
    
    submission_file = sys.argv[1]
    evaluate(submission_file)
