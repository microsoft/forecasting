# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains important path variables shared by all scripts
in the OrangeJuice_Pt_3Weeks_Weekly benchmark folder. It inserts
the TSPerf directory into sys.path, so that scripts can import
all the modules in TSPerf.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
TSPERF_DIR = os.path.dirname(os.path.dirname(BENCHMARK_DIR))

SUBMISSIONS_DIR = os.path.join(BENCHMARK_DIR, "submissions")
DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
