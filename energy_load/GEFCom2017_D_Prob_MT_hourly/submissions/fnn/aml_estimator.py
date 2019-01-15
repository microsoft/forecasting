import subprocess
import os
import sys
import getopt
import pandas as pd
from datetime import datetime
from azureml.core import Run
import time
start_time = time.time()
run = Run.get_submitted_run()

base_command = "Rscript train_validate_aml.R"

if __name__ == '__main__':
    opts, args = getopt.getopt(
        sys.argv[1:], '', ['path=', 'cv_path=', 'n_hidden_1=', 'n_hidden_2=',
                           'iter_max=', 'penalty='])
    for opt, arg in opts:
        if opt == '--path':
            path = arg
        elif opt == '--cv_path':
            cv_path = arg
        elif opt == '--n_hidden_1':
            n_hidden_1 = arg
        elif opt == '--n_hidden_2':
            n_hidden_2 = arg
        elif opt == '--iter_max':
            iter_max = arg
        elif opt == '--penalty':
            penalty = arg
<<<<<<< HEAD
    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
=======
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
>>>>>>> d2baee228c220622209b577dedb8c19f293bb14a
    task = " ".join([base_command,
                    '--path', path,
                    '--cv_path', cv_path,
                    '--n_hidden_1', n_hidden_1,
                    '--n_hidden_2', n_hidden_2,
                    '--iter_max', iter_max,
                    '--penalty', penalty,
                    '--time_stamp', time_stamp])
    process = subprocess.call(task, shell=True)

    # process.communicate()
    # process.wait()
    
    output_file_name = 'cv_output_' + time_stamp + '.csv'
    result = pd.read_csv(os.path.join(cv_path, output_file_name))

    APL = result['loss'].mean()

    print(APL)
    print("--- %s seconds ---" % (time.time() - start_time))

    run.log('average pinball loss', APL)
