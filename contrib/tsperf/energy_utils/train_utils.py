# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from fclib.feature_engineering.utils import add_datetime


class TSCVSplitter:
    """
        Creates cross validation time ranges for given back test configuration.

        Attributes:
            datetime_format(str): The format of all attributes that are
                datetime strings.
            train_start_time(str): The start time of the training data.
            backtest_start_time(datetime): The start time of the first-fold
                validation data
            backtest_step_size(str): The number of backtest_step_units
                between the validation start times of consecutive cross
                validation folds.
            backtest_step_unit(str): The time unit used to determine the time
                interval between the validation start times of consecutive
                cross validation folds. Valid values are 'year', 'month',
                'week', 'day', 'hour', 'minute'.
            backtest_end_time(datetime): The end time of the last-fold
                validation data
            validation_steps(int): The number of backtest_step_units to
                validate on in each cross validation fold.
            data_frequency(str): The frequency of the training and validation
                data. Valid values are 'year', 'month', 'week', 'day', 'hour',
                'minute'.
            train_validation_split(dict): The result dictionary defining the
                training and validation time ranges of each cross validation
                folds. Below is an example of a two-fold cross validation
                {
                 "cv_round_1": {
                   "train_range": [
                    "2011-01-01 00:00:00",
                    "2012-11-30 23:00:00"
                   ],
                   "validation_range": [
                    "2013-01-01 00:00:00",
                    "2013-01-31 23:00:00"
                   ]
                  },

                 "cv_round_2": {
                   "train_range": [
                    "2011-01-01 00:00:00",
                    "2013-11-30 23:00:00"
                   ],
                   "validation_range": [
                    "2014-01-01 00:00:00",
                    "2014-01-31 23:00:00"
                   ]
                  }
                }
    """

    def __init__(self, config):
        """
            Creates cross validation time ranges for given back test
            configuration.

            Args:
                config(dict): A dictionary defining how the back testing
                    should be performed. The config dictionary should follow
                    the following structure:
                    {
                      "DatetimeFormat": "%Y-%m-%d %H:%M:%S",
                      "DataFrequency": "hour",
                      "BackTestParams": {
                        "TrainStartTime": "2011-01-01 00:00:00",
                        "ValidationStartTime": "2013-01-01 00:00:00",
                        "StepSize": 1,
                        "StepUnit": "year",
                        "ValidationSteps": 1,
                        "EndTime": "2016-12-31 23:00:00",
                      }
                    }

        """
        self.datetime_format = config["DatetimeFormat"]

        back_test_config = config["BackTestParams"]
        self.train_start_time = back_test_config["TrainStartTime"]
        self.backtest_start_time = datetime.strptime(back_test_config["ValidationStartTime"], self.datetime_format)
        self.backtest_step_size = back_test_config["StepSize"]
        self.backtest_step_unit = back_test_config["StepUnit"]
        self.backtest_end_time = datetime.strptime(back_test_config["EndTime"], self.datetime_format)
        self.validation_steps = back_test_config["ValidationSteps"]

        self.data_frequency = config["DataFrequency"]

        self.train_validation_split = self.create_train_validation_split()

    def create_train_validation_split(self):
        """
            Creates cross validation time ranges for given back test
            configuration.
        """
        split_datetime_list = []
        split_datetime = self.backtest_start_time

        while split_datetime < self.backtest_end_time:
            split_datetime_list.append(split_datetime)
            split_datetime = add_datetime(split_datetime, self.backtest_step_unit, self.backtest_step_size)

        train_validation_split = {}
        train_start = self.train_start_time

        for i in range(len(split_datetime_list) - 1):
            validation_start = split_datetime_list[i]
            train_end = add_datetime(validation_start, self.data_frequency, -1)
            validation_end = add_datetime(train_end, self.backtest_step_unit, self.validation_steps)

            validation_start = validation_start.strftime(self.datetime_format)
            validation_end = validation_end.strftime(self.datetime_format)
            train_end = train_end.strftime(self.datetime_format)

            train_validation_split["cv_round_" + str(i + 1)] = {
                "train_range": [train_start, train_end],
                "validation_range": [validation_start, validation_end],
            }

        validation_start = split_datetime_list[-1]
        train_end = add_datetime(split_datetime_list[-1], self.data_frequency, -1)
        validation_end = self.backtest_end_time

        validation_start = validation_start.strftime(self.datetime_format)
        validation_end = validation_end.strftime(self.datetime_format)
        train_end = train_end.strftime(self.datetime_format)

        train_validation_split["cv_round_" + str(len(split_datetime_list))] = {
            "train_range": [train_start, train_end],
            "validation_range": [validation_start, validation_end],
        }

        return train_validation_split
