import os
import pandas as pd
from urllib.request import urlretrieve
from benchmark_settings import TRAIN_BASE_END, TRAIN_ROUNDS_ENDS, \
    TEST_STARTS_ENDS
from benchmark_paths import DATA_DIR
from datetime import time
    
# These columns should be dropped in the test data as they are not available
# at forecasting time
DROP_COLUMNS = ['DEMAND', 'DewPnt', 'DryBulb']


def split_train_test(full_df, output_dir,
                     train_base_file='train_base.csv',
                     train_file_prefix='train_round_',
                     test_file_prefix='test_round_'):
    """
    This helper function splits full_df into train and test folds as defined
    in benchmark_settings.py
    :param full_df:
        Full data frame to be split. For robustness and
        simplicity, it's required that full_df is indexed by datetime at
        level 0.
    :param output_dir: Directory to store the output files in.
    :param datetime_colname: Column or index in full_df used for splitting.
    :param train_base_file:
        This file stores all the data before TRAIN_BASE_END.
    :param train_file_prefix:
        These files each contains a subset of data to be added to
        train_base_file to form the training data of a particular round.
    :param test_file_prefix:
        These files each contains testing data for a particular round
    """

    train_data_dir = os.path.join(output_dir, 'train')
    test_data_dir = os.path.join(output_dir, 'test')
    ground_truth_dir = os.path.join(output_dir, 'test_ground_truth')

    # Create train and test data directories
    if not os.path.isdir(train_data_dir):
        os.mkdir(train_data_dir)

    if not os.path.isdir(test_data_dir):
        os.mkdir(test_data_dir)

    if not os.path.isdir(ground_truth_dir):
        os.mkdir(ground_truth_dir)

    index_value = full_df.index.get_level_values(0)
    train_base_df = full_df.loc[index_value < TRAIN_BASE_END].copy()

    train_base_df.to_csv(os.path.join(train_data_dir, train_base_file))
    print('Base training data frame size: {}'.format(train_base_df.shape))

    for i in range(len(TRAIN_ROUNDS_ENDS)):
        file_name = os.path.join(train_data_dir,
                                 train_file_prefix + str(i+1) + '.csv')
        train_round_delta_df = full_df.loc[
            (index_value >= TRAIN_BASE_END)
            & (index_value < TRAIN_ROUNDS_ENDS[i])]
        print('Round {0} additional training data size: {1}'
              .format(i+1, train_round_delta_df.shape))
        print('Minimum timestamp: {0}'
              .format(min(train_round_delta_df.index.get_level_values(0))))
        print('Maximum timestamp: {0}'
              .format(max(train_round_delta_df.index.get_level_values(0))))
        print('')
        train_round_delta_df.to_csv(file_name)

    for i in range(len(TEST_STARTS_ENDS)):
        test_file = os.path.join(test_data_dir,
                                 test_file_prefix + str(i+1) + '.csv')
        ground_truth_file = os.path.join(ground_truth_dir,
                                         test_file_prefix + str(i+1) + '.csv')
        start_end = TEST_STARTS_ENDS[i]
        test_round_df = full_df.loc[
            ((index_value >= start_end[0]) & (index_value < start_end[1]))
        ].copy()

        test_round_df.to_csv(ground_truth_file)
        test_round_df.drop(DROP_COLUMNS, inplace=True, axis=1)
        test_round_df.to_csv(test_file)

        print('Round {0} testing data size: {1}'
              .format(i+1, test_round_df.shape))
        print('Minimum timestamp: {0}'.format(min(
            test_round_df.index.get_level_values(0))))
        print('Maximum timestamp: {0}'.format(max(
            test_round_df.index.get_level_values(0))))
        print('')


def download_eei_data():
    '''
    Download hourly system load data for the New England bulk power system provided in the 
    industry-standard Edison Electric Institute (EEI) format.
    '''

    urls = [
        "https://www.iso-ne.com/static-assets/documents/2018/02/2018_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/2017/02/2017_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/2016/02/2016_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/2015/02/2015_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2014_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2013_eei_loads.txt", 
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2012_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2011_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2010_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2009_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2008_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2007_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2006_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2005_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2004_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2003_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2002_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2001_eei_loads.txt",
        "https://www.iso-ne.com/static-assets/documents/markets/hstdata/hourly/syslds_eei/2000_eei_loads.txt"
    ]

    for url in urls:

        url_tokens = url.split('/')
        fname = url_tokens[-1]
        fname

        fpath = os.path.join(DATA_DIR, fname)

        # Check if file already exists and skip if so
        if os.path.exists(fpath):
            print(fpath + " already exists")
            continue

        print('Downloading', url)

        f, _ = urlretrieve(url, fpath)
        print('Downloaded to', fpath)
        


def parse_eei_date(dt):
    """Helper function to parse date column from EEI format files."""
    from datetime import datetime
    dt = dt[:-1] # remove the last char (indicators 1 and 2)
    dt = dt.replace(' ', '0')
    if(len(dt) == 6):
        dt = datetime.strptime(dt, "%m%d%y")
    elif(len(dt) == 8):
        dt = datetime.strptime(dt, "%m%d%Y")   
    return(dt)


def parse_eei_load(ldstr):
    """Helper function to parse load column from EEI format files."""
    load = [ldstr[0+i:5+i] for i in range(0, len(ldstr), 5)]
    try:
        load = [int(l) for l in load]
    except TypeError:
        print('Load variable cannot be mapped to integer value.')
    return(load)


def extract_eei_data():
    """
    This function requires the user to use the script
    "TSPerf/energy_load/GEFCom2017-D_Prob_MT_hourly/common/download_eei_data.py" to
    download the EEI Hourly Data from 2008 to 2018 from the ISO New England
    website (https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/sys-load-eei-fmt).
    The downloaded data is stored in
    "TSPerf/energy_load/TSPerf/energy_load/GEFCom2017-D_Prob_MT_hourly/data"

    This script parses the txt files and creates a csv file for each original txt file.

    The output files contain the following columns
    Datetime:
        Generated by combining the Date columns of the csv files and appending 
        hour stamp to every day.
    DEMAND:
        Real-Time Demand is Non-PTF Demand for wholesale market settlement
        from revenue quality metering, and is defined as the sum of non-dispatchable
        load assets, station service load assets, and unmetered load assets. This load
        corresponds to total load across all zones.
    """
    
    # Look for files from 2000 to 2018
    start_yr = 2000
    end_yr = 2018
    yrs = list(range(start_yr, end_yr+1))
    files = [str(y)+"_eei_loads.txt" for y in yrs]

    for f in files:

        f = os.path.join(DATA_DIR, f)
        try:
            with open(f) as file:  
                data = file.readlines()
        except FileNotFoundError:
            raise Exception('The data file {0} is not found in the data '
                            'directory {1}, make sure you download the data '
                            'and try again.'.format(f, DATA_DIR))

        data = [l.replace('\n', '') for l in data]

        # collect dates into a list
        dates = [l.split('  ')[0] for l in data]
        dates = [parse_eei_date(dt) for dt in dates]

        # remove extra lines with next year
        yrs = [d.year for d in dates]
        this_year = max(set(yrs), key = yrs.count)
        dates = [x for x in dates if x.year == this_year]

        # check that every two consecutive dates are equal
        assert dates[0::2] == dates[1::2]

        # collect energy loads into a list
        loads = [l[20:90] for l in data]
        loads = [parse_eei_load(l) for l in loads]
        loads = [x for x in loads if x] # remove empty trailing lines

        # combine dates and loads into 24 hour long vectors
        dates24 = dates[0::2]

        # create 24 hours list
        hours24 = list(range(24))
        hours24 = [time(h) for h in hours24]

        loads1 = loads[0::2]
        loads2 = loads[1::2]
        loads_24 = []
        for i in range(len(loads1)):
            loads24 = loads1[i] + loads2[i]
            my_df = pd.DataFrame({'hour': hours24, 'DEMAND': loads24}, columns=['hour', 'DEMAND'])
            loads_24.append(my_df)

        loads_df = pd.concat(loads_24, ignore_index=True)

        # repeat dates 24 times and concat to the data frame
        dates24_hr = [item for item in dates24 for i in range(24)]

        loads_df['date'] = dates24_hr

        format_date = lambda x: str(x.date())
        loads_df['Datetime'] = loads_df['date'].map(format_date) + ' ' + loads_df['hour'].map(str)
        loads_df = loads_df[['Datetime', 'DEMAND']]

        # write data to csv file 
        csv_file = os.path.basename(os.path.splitext(f)[0]) + '.csv'
        csv_file = os.path.join(DATA_DIR, csv_file)
        loads_df.to_csv(csv_file, index=False)
    
