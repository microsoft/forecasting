from datetime import datetime
from common.utils import get_offset_by_frequency


class TSCVSplitter:
    """
    Creates cross validation time ranges for given back test configuration.

    Args:
        back_test_config(dict): Configuration of the back testing procedure,
            should contain the following key-value pairs:
            "train_start_time": str, start time of the training data
            available for back testing.
            "train_end_time": str, end time of the training data available
            for back testing.
            "cv_folds": int, number of cross validation folds to perform
            "validation_steps": int, number of time steps, in the unit of
                validation_step_unit, to validate on in each cross
                validation fold
            "validation_step_unit": str, optional. Unit of
                validation_validation stride and validation_steps.
                Supported values are frequency strings corresponding to
                pandas.tseries.offsets.
                See https://pandas.pydata.org/pandas-docs/stable/user_guide/
                timeseries.html#dateoffset-objects
                Default value is set to the 'frequency' in df_config.
            "validation_stride": int, optional. Number of
                validatoin_step_units between the validation start times of
                consecutive cross validation folds. Default value is set to
                equal to validation_steps
            "fixed_train_size": bool, optional. If keep the training data
                sizes of different cross validation folds the same. If
                true, the training start time of each fold is increased from
                that of the previous fold to maintain a fixed training data
                size. Default value is false.
            "train_validation_gap": str, optional. Number of time steps,
                in the unit of train_validation_gap_unit, between the training
                end time and validation start time of each cv fold. Default
                value is set to 1.
            "train_validation_gap_unit": str, optional. Unit of
                train_validation_gap. Supported values are frequency strings
                corresponding to pandas.tseries.offsets.
                See https://pandas.pydata.org/pandas-docs/stable/user_guide/
                timeseries.html#dateoffset-objects
                Default value is set to the 'frequency' in df_config.
        df_config(dict): Configuration of the data to perform back testing
            on, should contain the following key-value paris:
            "frequency": str, frequency strings corresponding to
                pandas.tseries.offsets.
                See https://pandas.pydata.org/pandas-docs/stable/user_guide/
                timeseries.html#dateoffset-objects
            "time_format": Format of the timestamps. See http://strftime.org/.
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

    def __init__(self, back_test_config, df_config):

        self.data_frequency = df_config["frequency"]
        self.time_format = df_config["time_format"]
        self._offset = get_offset_by_frequency(self.data_frequency)

        self.train_start_time = back_test_config["train_start_time"]
        self.train_end_time = back_test_config["train_end_time"]
        self.cv_folds = back_test_config["cv_folds"]
        self.validation_steps = back_test_config["validation_steps"]

        if "validation_stride" not in back_test_config:
            self.validation_stride = self.validation_steps
        else:
            self.validation_stride = back_test_config["validation_stride"]

        if "validation_step_unit" not in back_test_config:
            self.validation_step_unit = self.data_frequency
        else:
            self.validation_step_unit = back_test_config[
                "validation_step_unit"
            ]

        if "train_validation_gap" not in back_test_config:
            self.train_validation_gap = 1
        else:
            self.train_validation_gap = back_test_config[
                "train_validation_gap"
            ]

        if "train_validation_gap_unit" not in back_test_config:
            self.train_validation_gap_unit = self.data_frequency
        else:
            self.train_validation_gap_unit = back_test_config[
                "train_validation_gap_unit"
            ]

        if "fixed_train_size" not in back_test_config:
            self.fixed_train_size = False
        elif isinstance(back_test_config["fixed_train_size"], bool):
            self.fixed_train_size = back_test_config["fixed_train_size"]
        else:
            raise ValueError("'fixed_train_size' must be boolean")

        self._round_name_prefix = "cv_round_"

        self.train_validation_split = self.create_train_validation_split()

    def _compute_train_size(self, train_validation_split, round_number):
        round_split = train_validation_split[
            self._round_name_prefix + str(round_number)
        ]
        train_size = (
            round_split["train_range"][1] - round_split["train_range"][0]
        )

        return train_size

    def create_train_validation_split(self):
        """
            Creates cross validation time ranges for given back test
            configuration.
        """
        validation_step_offset = get_offset_by_frequency(
            self.validation_step_unit
        )
        train_validation_gap_offset = get_offset_by_frequency(
            self.train_validation_gap_unit
        )

        train_validation_split = {}
        train_start = datetime.strptime(
            self.train_start_time, self.time_format
        )
        validation_end = datetime.strptime(
            self.train_end_time, self.time_format
        )
        for iR in range(self.cv_folds):
            fold_number = self.cv_folds - iR
            validation_start = (
                validation_end - self.validation_steps * validation_step_offset
            )

            train_end = (
                validation_start
                - self.train_validation_gap * train_validation_gap_offset
            )

            if train_end < train_start:
                raise Exception(
                    "Not enough data to perform {} folds of "
                    "cross validation. Please reduce cv_folds "
                    "validation_stride, "
                    "or validation_steps".format(self.cv_folds)
                )
            train_validation_split[
                self._round_name_prefix + str(fold_number)
            ] = {
                "train_range": [train_start, train_end],
                "validation_range": [validation_start, validation_end],
            }

            # Update validation_end for the next fold
            validation_end = (
                validation_end
                - self.validation_stride * validation_step_offset
            )

        if self.fixed_train_size is True:
            first_round_train_size = self._compute_train_size(
                train_validation_split, 1
            )
            last_round_train_size = self._compute_train_size(
                train_validation_split, self.cv_folds
            )
            if first_round_train_size < last_round_train_size:
                for iR in range(2, self.cv_folds + 1):
                    round_train_size = self._compute_train_size(
                        train_validation_split, iR
                    )
                    if round_train_size > first_round_train_size:
                        round_name = self._round_name_prefix + str(iR)
                        train_end = train_validation_split[round_name][
                            "train_range"
                        ][1]
                        train_start_new = train_end - first_round_train_size
                        train_validation_split[round_name]["train_range"][
                            0
                        ] = train_start_new

        for iR in range(1, self.cv_folds + 1):
            round_name = self._round_name_prefix + str(iR)
            train_validation_split[round_name]["train_range"][
                0
            ] = train_validation_split[round_name]["train_range"][0].strftime(
                self.time_format
            )
            train_validation_split[round_name]["train_range"][
                1
            ] = train_validation_split[round_name]["train_range"][1].strftime(
                self.time_format
            )
            train_validation_split[round_name]["validation_range"][
                0
            ] = train_validation_split[round_name]["validation_range"][
                0
            ].strftime(
                self.time_format
            )
            train_validation_split[round_name]["validation_range"][
                1
            ] = train_validation_split[round_name]["validation_range"][
                1
            ].strftime(
                self.time_format
            )

        return train_validation_split
