"""
This file contains training and testing settings to be used in this benchmark, mainly:
    TRAIN_BASE_END: Base training end date common across all rounds
    TRAIN_ROUNDS_ENDS: a set of dates denoting end of trianing period for each of the 6 rounds of the benchmark
    TEST_STARTS_ENDS: a set of dates denoting start and end of testing period for each of the 6 rounds of the benchmark
"""

import pandas as pd

TRAIN_BASE_END = pd.to_datetime('2016-11-01')
TRAIN_ROUNDS_ENDS = pd.to_datetime(['2016-12-01', '2016-12-01',
                                    '2017-01-01', '2017-01-01',
                                    '2017-02-01', '2017-02-01'])

TEST_STARTS_ENDS = [pd.to_datetime(('2017-01-01', '2017-02-01')),
                    pd.to_datetime(('2017-02-01', '2017-03-01')),
                    pd.to_datetime(('2017-02-01', '2017-03-01')),
                    pd.to_datetime(('2017-03-01', '2017-04-01')),
                    pd.to_datetime(('2017-03-01', '2017-04-01')),
                    pd.to_datetime(('2017-04-01', '2017-05-01'))]