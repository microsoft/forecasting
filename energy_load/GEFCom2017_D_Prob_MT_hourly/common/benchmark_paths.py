"""
This file contains important path variables shared by all scripts
in the GEFCom2017_D_Prob_MT_hourly benchmark folder. It inserts
the TSPerf directory into sys.path, so that scripts can import
all the modules in TSPerf.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
TSPERF_DIR = os.path.dirname(os.path.dirname(BENCHMARK_DIR))

SUBMISSIONS_DIR = os.path.join(BENCHMARK_DIR, 'submissions')
DATA_DIR = os.path.join(BENCHMARK_DIR, 'data')
HOLIDAY_DATA_PATH = os.path.join(TSPERF_DIR, 'common', 'us_holidays.csv')

if TSPERF_DIR not in sys.path:
    sys.path.insert(0, TSPERF_DIR)
