import json
import os
import itertools
from datetime import datetime
from dateutil.relativedelta import relativedelta
import subprocess

from common.train_utils import CrossValidator


class ParameterSweeper:
    """
        The function of this class is currently replaced by HyperDrive.
        But let's keep it to preserve the work already done, and also
        in case we need more flexibility than what HyperDrive provides.
    """

    def __init__(self, config):

        self.work_directory = config['WorkDirectory']
        data_config = config['DataParams']
        self.data_path = data_config['DataPath']
        if 'DataFile' in data_config:
            data_file = data_config['DataFile']
            self.data_full_path = os.path.join(self.work_directory,
                                               self.data_path, data_file)
        else:
            self.data_full_path = os.path.join(self.work_directory,
                                               self.data_path)

        parameters_config = config['Parameters']
        self.parameter_name_list = [n for n, _ in parameters_config.items()]
        parameter_value_list = [p for _, p in parameters_config.items()]
        self.parameter_combinations = \
            list(itertools.product(*parameter_value_list))

        features_config = config['Features']
        self.feature_selection_mode = features_config['FeatureSelectionMode']
        if self.feature_selection_mode == 'Default':
            # In default mode, simply iterate through each feature set in
            # FeatureList
            self.feature_list = features_config['FeatureList']
        else:
            # Placeholder for more advanced feature selection strategy
            pass

    def sweep_parameters_script(self, script_config,
                                cv_setting_file, params_setting_file):
        script_command = script_config['ScriptCommand']
        script = os.path.join(self.work_directory, script_config['Script'])
        task_list = []
        parameter_sets = {}
        count = 0
        for f in self.feature_list:
            for p in self.parameter_combinations:
                count += 1
                task = ' '.join([script_command, script,
                                 '-d', self.data_full_path,
                                 '-p', params_setting_file,
                                 '-c', cv_setting_file,
                                 '-s', str(count)])
                task_list.append(task)

                parameter_dict = {}

                for n, v in zip(self.parameter_name_list, p):
                    parameter_dict[n] = v

                parameter_sets[count] = {'feature_set': f,
                                         'features': self.feature_list[f],
                                         'parameters': parameter_dict}
        with open(params_setting_file, 'w') as fp:
            json.dump(parameter_sets, fp, indent=True)

        # Run tasks in parallel
        processes = []

        for t in task_list:
            process = subprocess.Popen(t, shell=True)
            processes.append(process)

        # Collect statuses
        output = [p.wait() for p in processes]

        print(output)

    def sweep_parameters(self):
        # placeholder for parameter sweeping in python
        pass

    def sweep_parameters_batch_ai(self):
        # placeholder for parameter sweeping using batch ai
        pass


def main(config_file):
    with open(config_file) as f:
        config = json.load(f)

    datetime_format = config['DatetimeFormat']
    work_directory = config['WorkDirectory']

    cv_setting_file = os.path.join(work_directory, 'cv_settings.json')
    # parameter_setting_file = os.path.join(work_directory,
    #                                       'parameter_settings.json')

    cv = CrossValidator(config)

    # This part adjusts the cv settings due to the specific problem setup
    # of GEFCom2017. Different forecasting setups may require different
    # adjustments. Most setups should not require any adjustment.
    for k, v in cv.train_validation_split.items():
        round_dict = {}
        # Training data ends on 12/31, used to forecast Feb. and Mar.
        train_end = datetime.strptime(v['train_range'][1], datetime_format)

        # Jan. validation range
        validation_start_1 = datetime.strptime(v['validation_range'][0],
                                               datetime_format)
        validation_end_1 = validation_start_1 + \
                           relativedelta(months=1, hours=-1)

        # Training data ends on 11/30, used to forecast Jan. and Feb.
        train_end_prev = datetime.strftime(
            train_end + relativedelta(months=-1), datetime_format)
        # Training data ends on 01/31, used to forecast Mar. and Apr.
        train_end_next = datetime.strftime(
            train_end + relativedelta(months=1), datetime_format)

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

        round_dict[1] = {'train_range': [v['train_range'][0], train_end_prev],
                         'validation_range': [validation_start_1, validation_end_1]
                         }
        round_dict[2] = {'train_range': [v['train_range'][0], train_end_prev],
                         'validation_range': [validation_start_2, validation_end_2]
                         }
        round_dict[3] = {'train_range': [v['train_range'][0], v['train_range'][1]],
                         'validation_range': [validation_start_2, validation_end_2]
                         }
        round_dict[4] = {'train_range': [v['train_range'][0], v['train_range'][1]],
                         'validation_range': [validation_start_3, validation_end_3]
                         }

        round_dict[5] = {'train_range': [v['train_range'][0], train_end_next],
                         'validation_range': [validation_start_3, validation_end_3]
                         }
        round_dict[6] = {'train_range': [v['train_range'][0], train_end_next],
                         'validation_range': [validation_start_4, validation_end_4]
                         }

        cv.train_validation_split[k] = round_dict

    with open(cv_setting_file, 'w') as fp:
        json.dump(cv.train_validation_split, fp, indent=True)
    #
    # ps = ParameterSweeper(config)
    #
    # script_config = config['ScriptParams']
    # ps.sweep_parameters_script(script_config, cv_setting_file,
    #                            parameter_setting_file)


if __name__ == '__main__':
    main('backtest_config.json')

