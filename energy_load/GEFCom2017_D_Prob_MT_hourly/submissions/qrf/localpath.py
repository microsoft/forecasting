"""
This script inserts the TSPerf directory into sys.path, so that scripts can import
all the modules in TSPerf. Each submission folder needs its own localpath.py file.
"""

import os, sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSIONS_DIR = os.path.dirname(CUR_DIR)
BENCHMARK_DIR = os.path.dirname(SUBMISSIONS_DIR)
TSPERF_DIR = os.path.dirname(os.path.dirname(BENCHMARK_DIR))
COMMON_DIR = os.path.join(TSPERF_DIR,'common')

if TSPERF_DIR not in sys.path:
    sys.path.insert(0, TSPERF_DIR)
    sys.path.insert(0, COMMON_DIR)
