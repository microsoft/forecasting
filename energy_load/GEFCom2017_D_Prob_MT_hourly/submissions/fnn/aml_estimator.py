import subprocess
import os
import sys
import getopt
import pandas as pd
from azureml.core import Run
run = Run.get_submitted_run()

base_command = "Rscript train_validate_aml.R"

if __name__ == '__main__':
    opts, args = getopt.getopt(
        sys.argv[1:], '', ['path=', 'n_hidden_1=', 'n_hidden_2=',
                           'iter_max=', 'penalty='])
    for opt, arg in opts:
        if opt == '--path':
            path = arg
        elif opt == '--n_hidden_1':
            n_hidden_1 = arg
        elif opt == '--n_hidden_2':
            n_hidden_2 = arg
        elif opt == '--iter_max':
            iter_max = arg
        elif opt == '--penalty':
            penalty = arg
    task = " ".join([base_command,
                    '--path', path,
                    '--n_hidden_1', n_hidden_1,
                    '--n_hidden_2', n_hidden_2,
                    '--iter_max', iter_max,
                    '--penalty', penalty])
    process = subprocess.call(task, shell=True)

    # process.communicate()
    # process.wait()

    result = pd.read_csv(os.path.join(path, 'cv_output.csv'))

    APL = result['loss'].mean()

    run.log('average pinball loss', APL)
