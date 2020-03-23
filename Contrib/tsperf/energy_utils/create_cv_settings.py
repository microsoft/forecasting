# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script takes a backtest configuration file as input and generates a cross
validation setting file containing training and validation time ranges for
each cross validation round. See the backtest_config.json file and
cv_settings.json file in TSPerf/prototypes/cross_validation for examples
of the configuration files.

There are two levels of cross validation folds.
The first level is year and here are the yearly training folds
cv_round_1: training data: 2011 - 2012, validation data: 2013
cv_round_2: training data: 2011 - 2013, validation data: 2014
cv_round_3: training data: 2011 - 2014, validation data: 2015
cv_round_4: training data: 2011 - 2015, validation data: 2016
The second level is within each year. Two options are provided and can be
configured by the "FoldsPerYear" field in the back test configuration file.
Two values are supported for FoldsPerYear: 6 and 12.
When FoldsPerYear = 6, the within-year folds are split based on the set up
of GEFCom2017, which can be found in this link.
http://blog.drhongtao.com/2016/10/instructions-for-gefcom2017-qualifying-match.html
When FoldsPerYear = 12, the within-year folds are created by shifting a
one-month validation window 8 days at a time, between Jan. 1st and Apr.
30th. This approach is based on a GEFCom2017 winning solution from a
Microsoft team.

For examples of doing cross validation and parameter tuning with the output
of this script, see TSPerf/prototypes/cross_validation.

"""

import json
import os
import sys
import getopt
from datetime import datetime
from dateutil.relativedelta import relativedelta

from fclib.evaluation.train_utils import TSCVSplitter


def main(config_file):
    """
    Main function that takes backtest configuration file as input and
    generates a cross-validation settings file containing training and
    validation time ranges for each cross validation round.
    """

    with open(config_file) as f:
        config = json.load(f)

    datetime_format = config["DatetimeFormat"]
    work_directory = config["WorkDirectory"]
    cv_setting_file = os.path.join(work_directory, config["BackTestParams"]["CVSettingFile"])
    folds_per_year = config["BackTestParams"]["FoldsPerYear"]

    if folds_per_year not in (6, 12):
        raise Exception(
            "Invalid folds_per_year value, {}, provided." "Valid values are 6 and 12".format(folds_per_year)
        )

    cv = TSCVSplitter(config)

    if folds_per_year == 6:
        # This part adjusts the cv settings based on the specific problem setup
        # of GEFCom2017.
        for k, v in cv.train_validation_split.items():
            round_dict = {}
            # Training data ends on 12/31, used to forecast Feb. and Mar.
            train_end = datetime.strptime(v["train_range"][1], datetime_format)

            # Jan. validation range
            validation_start_1 = datetime.strptime(v["validation_range"][0], datetime_format)
            validation_end_1 = validation_start_1 + relativedelta(months=1, hours=-1)

            # Training data ends on 11/30, used to forecast Jan. and Feb.
            train_end_prev = datetime.strftime(train_end + relativedelta(months=-1), datetime_format)
            # Training data ends on 01/31, used to forecast Mar. and Apr.
            train_end_next = datetime.strftime(train_end + relativedelta(months=1), datetime_format)

            # Feb. validation range
            validation_start_2 = validation_start_1 + relativedelta(months=1)
            validation_end_2 = validation_start_2 + relativedelta(months=1, hours=-1)

            # Mar. validation range
            validation_start_3 = validation_start_1 + relativedelta(months=2)
            validation_end_3 = validation_start_3 + relativedelta(months=1, hours=-1)

            # Apr. validation range
            validation_start_4 = validation_start_1 + relativedelta(months=3)
            validation_end_4 = validation_start_4 + relativedelta(months=1, hours=-1)

            validation_start_1 = datetime.strftime(validation_start_1, datetime_format)
            validation_end_1 = datetime.strftime(validation_end_1, datetime_format)
            validation_start_2 = datetime.strftime(validation_start_2, datetime_format)
            validation_end_2 = datetime.strftime(validation_end_2, datetime_format)
            validation_start_3 = datetime.strftime(validation_start_3, datetime_format)
            validation_end_3 = datetime.strftime(validation_end_3, datetime_format)
            validation_start_4 = datetime.strftime(validation_start_4, datetime_format)
            validation_end_4 = datetime.strftime(validation_end_4, datetime_format)

            round_dict[1] = {
                "train_range": [v["train_range"][0], train_end_prev],
                "validation_range": [validation_start_1, validation_end_1],
            }
            round_dict[2] = {
                "train_range": [v["train_range"][0], train_end_prev],
                "validation_range": [validation_start_2, validation_end_2],
            }
            round_dict[3] = {
                "train_range": [v["train_range"][0], v["train_range"][1]],
                "validation_range": [validation_start_2, validation_end_2],
            }
            round_dict[4] = {
                "train_range": [v["train_range"][0], v["train_range"][1]],
                "validation_range": [validation_start_3, validation_end_3],
            }

            round_dict[5] = {
                "train_range": [v["train_range"][0], train_end_next],
                "validation_range": [validation_start_3, validation_end_3],
            }
            round_dict[6] = {
                "train_range": [v["train_range"][0], train_end_next],
                "validation_range": [validation_start_4, validation_end_4],
            }

            cv.train_validation_split[k] = round_dict
    elif folds_per_year == 12:
        # This part adjusts the cv settings based on a winning solution from
        # a Microsoft team
        for k, v in cv.train_validation_split.items():
            train_start = v["train_range"][0]
            validation_fold_start = datetime.strptime(v["validation_range"][0], datetime_format)
            validation_fold_end = validation_fold_start + relativedelta(months=1, hours=-1)
            validation_end = validation_fold_start + relativedelta(months=4, hours=-1)
            iF = 1
            round_dict = {}
            while validation_fold_end <= validation_end:
                train_fold_end = validation_fold_start + relativedelta(weeks=-9, hours=-1)
                train_fold_end = datetime.strftime(train_fold_end, datetime_format)
                round_dict[iF] = {
                    "train_range": [train_start, train_fold_end],
                    "validation_range": [
                        datetime.strftime(validation_fold_start, datetime_format),
                        datetime.strftime(validation_fold_end, datetime_format),
                    ],
                }
                validation_fold_start = validation_fold_start + relativedelta(days=8)
                validation_fold_end = validation_fold_end + relativedelta(days=8)
                iF += 1

            cv.train_validation_split[k] = round_dict

    with open(cv_setting_file, "w") as fp:
        json.dump(cv.train_validation_split, fp, indent=True)


if __name__ == "__main__":
    config_file = "backtest_config.json"
    opts, args = getopt.getopt(sys.argv[1:], "", ["config_file="])
    for opt, arg in opts:
        if opt == "--config_file":
            config_file = arg
    main(config_file)
