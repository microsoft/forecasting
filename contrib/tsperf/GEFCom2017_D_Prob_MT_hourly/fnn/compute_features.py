"""
This script uses
energy_load/GEFCom2017_D_Prob_MT_hourly/common/feature_engineering.py to
compute a list of features needed by the Feed-forward Neural Network model.
"""

import os
import sys
import getopt

import localpath

from tsperf.benchmarking.GEFCom2017_D_Prob_MT_hourly.feature_engineering import compute_features

SUBMISSIONS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(SUBMISSIONS_DIR, "data")
print("Data directory used: {}".format(DATA_DIR))

OUTPUT_DIR = os.path.join(DATA_DIR, "features")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

DF_CONFIG = {
    "time_col_name": "Datetime",
    "ts_id_col_names": "Zone",
    "target_col_name": "DEMAND",
    "frequency": "H",
    "time_format": "%Y-%m-%d %H:%M:%S",
}

# Feature configuration list used to specify the features to be computed by
# compute_features.
# Each feature configuration is a tuple in the format of (feature_name,
# featurizer_args)
# feature_name is used to determine the featurizer to use, see FEATURE_MAP in
# energy_load/GEFCom2017_D_Prob_MT_hourly/common/feature_engineering.py
# featurizer_args is a dictionary of arguments passed to the
# featurizer
feature_config_list = [
    ("temporal", {"feature_list": ["hour_of_day", "month_of_year"]}),
    ("annual_fourier", {"n_harmonics": 3}),
    ("weekly_fourier", {"n_harmonics": 3}),
    ("previous_year_load_lag", {"input_col_names": "DEMAND", "round_agg_result": True},),
    ("previous_year_temp_lag", {"input_col_names": "DryBulb", "round_agg_result": True},),
]


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "", ["submission="])
    for opt, arg in opts:
        if opt == "--submission":
            submission_folder = arg
            output_data_dir = os.path.join(SUBMISSIONS_DIR, submission_folder, "data")
            if not os.path.isdir(output_data_dir):
                os.mkdir(output_data_dir)
            OUTPUT_DIR = os.path.join(output_data_dir, "features")
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    compute_features(
        TRAIN_DATA_DIR,
        TEST_DATA_DIR,
        OUTPUT_DIR,
        DF_CONFIG,
        feature_config_list,
        filter_by_month=True,
        compute_load_ratio=True,
    )
