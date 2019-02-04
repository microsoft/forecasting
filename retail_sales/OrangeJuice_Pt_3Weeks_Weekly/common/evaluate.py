"""
This script evaluates an implementation of the 
OrangeJuice_Pt_3Weeks_Weekly benchmark. It reads in
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
import numpy as np
sys.path.append('.')
from common.evaluation_utils import MAPE
from benchmark_settings import NUM_ROUNDS

def read_test_files(benchmark_dir):
    """Get the ground truth of the forecasts.
    
    Args:
        benchmark_dir (String): Directory of the benchmark

    Returns:
        Dataframe including the ground truth of all forecast rounds
    """
    test_data_dir = os.path.join(benchmark_dir, "data", "test")
    for rnd in range(1, NUM_ROUNDS+1):
        test_file = 'test_round_'+str(rnd)+'.csv'
        test_round = pd.read_csv(os.path.join(test_data_dir, test_file))
        test_round['round'] = rnd
        test_round = test_round[['round', 'store', 'brand', 'week', 'logmove']]
        if rnd > 1:
            test = test.append(test_round)
        else:
            test = test_round.copy()
    return test


def evaluate(submission_file):
    """Evaluate and print the quality of the forecast.

    Args:
        submission_file (String): Submission file name
    """
    benchmark_dir = os.path.join("retail_sales", "OrangeJuice_Pt_3Weeks_Weekly")

    test = read_test_files(benchmark_dir)

    submission = pd.read_csv(submission_file)

    evaluation = pd.merge(submission, test, on=['round', 'store', 'brand', 'week'], how='left')

    # convert log sales to sales
    evaluation['sales'] = evaluation['logmove'].apply(lambda x: round(np.exp(x)))

    print("MAPE: ", MAPE(evaluation['prediction'], evaluation['sales'])*100)



if __name__=="__main__":
    
    submission_file = sys.argv[1]
    evaluate(submission_file)