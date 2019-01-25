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
    submission_file: relative file path to submission.csv 
        to the reference or submission implementation
"""


import os
import sys
import pandas as pd
sys.path.append('.')
from benchmark_paths import BENCHMARK_DIR
from common.evaluation_utils import pinball_loss


def read_test_files(benchmark_dir):
    """Helper function to read test files for all rounds in the benchmark."""

    test_data_dir = os.path.join(benchmark_dir, "data", "test_ground_truth")
    for rnd in range(1, 7):
        test_file = 'test_round_'+str(rnd)+'.csv'
        test_round = pd.read_csv(os.path.join(test_data_dir, test_file), parse_dates=['Datetime'])
        test_round['Round'] = rnd
        test_round = test_round[['Round', 'Datetime', 'Zone', 'DEMAND']]
        if rnd > 1:
            test = test_round.append(test_round)
        else:
            test = test_round.copy()
    return test


def evaluate(submission_file):
    """
    Function that evaluates a submission file against test files. It prints out the pinball loss for each Zone in the benchmark, and the mean pinball loss across all Zones.

    Args:
        submission_file (str): relative path to the submission.csv file, that is the file containing predictions for the benchmark test data. 
    """

    test = read_test_files(BENCHMARK_DIR)

    print(os.path.join(BENCHMARK_DIR, submission_file))
    submission = pd.read_csv(os.path.join(BENCHMARK_DIR, submission_file), parse_dates=['Datetime'])

    evaluation = pd.merge(submission, test, on=['Round', 'Datetime', 'Zone'], how='left')

    evaluation['pinball'] = pinball_loss(evaluation['Prediction'], evaluation['DEMAND'], evaluation['q'])

    print("Mean pinball loss: ", evaluation['pinball'].mean())

    print(evaluation[['Zone', 'pinball']].groupby('Zone').mean().rename(columns={'pinball': 'mean pinball loss'}))
    

if __name__ == "__main__":
    submission_file = sys.argv[1]
    evaluate(submission_file)
