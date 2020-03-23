"""
This script inserts the TSPerf directory into sys.path, so that scripts can
import all the modules in TSPerf. Each submission folder needs its own
localpath.py file.
"""

import os, sys

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
TSPERF_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_CURR_DIR)))

if TSPERF_DIR not in sys.path:
    sys.path.insert(0, TSPERF_DIR)
