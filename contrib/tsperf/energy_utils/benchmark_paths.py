# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains important path variables shared by all scripts
in the GEFCom2017_D_Prob_MT_hourly benchmark folder. It inserts
the TSPerf directory into sys.path, so that scripts can import
all the modules in TSPerf.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_NAME = os.path.split(SCRIPT_DIR)[1]
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
BENCHMARK_DIR = os.path.join(REPO_DIR, "benchmarks", BENCHMARK_NAME)
DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
TSPERF_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
HOLIDAY_DATA_PATH = os.path.join(TSPERF_DIR, "feature_engineering", "us_holidays.csv")
