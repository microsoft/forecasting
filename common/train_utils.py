from datetime import datetime
from .utils import add_datetime


class TSCVSplitter:
    def __init__(self, config):
        self.datetime_format = config['DatetimeFormat']

        back_test_config = config['BackTestParams']
        self.train_start_time = back_test_config['TrainStartTime']
        self.backtest_start_time = datetime.strptime(
            back_test_config['ValidationStartTime'], self.datetime_format)
        self.backtest_step_size = back_test_config['StepSize']
        self.backtest_step_unit = back_test_config['StepUnit']
        self.backtest_end_time = datetime.strptime(
            back_test_config['EndTime'], self.datetime_format)
        self.validation_steps = back_test_config["ValidationSteps"]
        self.backtest_sliding_window = back_test_config['SlidingWindow']

        self.data_frequency = config['DataFrequency']

        self.train_validation_split = self.create_train_validation_split()

    def create_train_validation_split(self):
        split_datetime_list = []
        split_datetime = self.backtest_start_time

        while split_datetime < self.backtest_end_time:
            split_datetime_list.append(split_datetime)
            split_datetime = add_datetime(split_datetime,
                                          self.backtest_step_unit,
                                          self.backtest_step_size)

        train_validation_split = {}
        train_start = self.train_start_time

        for i in range(len(split_datetime_list) - 1):
            validation_start = split_datetime_list[i]
            train_end = add_datetime(validation_start, self.data_frequency, -1)
            validation_end = add_datetime(train_end,
                                          self.backtest_step_unit,
                                          self.validation_steps)

            validation_start = validation_start.strftime(self.datetime_format)
            validation_end = validation_end.strftime(self.datetime_format)
            train_end = train_end.strftime(self.datetime_format)

            train_validation_split['cv_round_' + str(i+1)] \
                = {'train_range': [train_start, train_end],
                   'validation_range': [validation_start, validation_end]}

        validation_start = split_datetime_list[-1]
        train_end = add_datetime(split_datetime_list[-1],
                                 self.data_frequency, -1)
        validation_end = self.backtest_end_time

        validation_start = validation_start.strftime(self.datetime_format)
        validation_end = validation_end.strftime(self.datetime_format)
        train_end = train_end.strftime(self.datetime_format)

        train_validation_split['cv_round_' + str(len(split_datetime_list))] \
            = {'train_range': [train_start, train_end],
               'validation_range': [validation_start, validation_end]}

        return train_validation_split
