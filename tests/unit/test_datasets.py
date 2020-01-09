import os
import sys
import pytest
import subprocess
import pandas as pd

def test_download_retail_data():
    BENCHMARK_DIR = os.path.join(".", "retail_sales", "OrangeJuice_Pt_3Weeks_Weekly")
    DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
    SCRIPT_PATH = os.path.join(BENCHMARK_DIR, "common", "download_data.r")
    DATA_FILE_LIST = ["yx.csv", "storedemo.csv"]
    # Remove data files if they are existed
    for f in DATA_FILE_LIST:
        file_path = os.path.join(DATA_DIR, f)
        if os.path.exists(file_path):
            os.remove(file_path)
        assert not os.path.exists(file_path)
    # Call data download script
    try:
        subprocess.call(["Rscript", SCRIPT_PATH])
    except subprocess.CalledProcessError as e:
        print(e.output)
    # Check downloaded data
    DATA_DIM_LIST = [(106139, 19), (83, 12)]
    COLUMN_NAME_LIST = [["store", "brand", "week", "logmove", "constant",
                          "price1", "price2", "price3", "price4", "price5",
                          "price6", "price7", "price8", "price9", "price10",
                          "price11", "deal", "feat", "profit"],
                         ["STORE", "AGE60", "EDUC", "ETHNIC", "INCOME",
                          "HHLARGE", "WORKWOM","HVAL150","SSTRDIST",
                          "SSTRVOL", "CPDIST5", "CPWVOL5"]
                        ]
    for idx, f in enumerate(DATA_FILE_LIST):
        file_path = os.path.join(DATA_DIR, f)
        assert os.path.exists(file_path) 
        df = pd.read_csv(file_path, index_col=None)
        assert df.shape == DATA_DIM_LIST[idx]
        assert list(df) == COLUMN_NAME_LIST[idx]

def test_download_energy_data():
    BENCHMARK_DIR = os.path.join(".", "energy_load", "GEFCom2017_D_Prob_MT_hourly")
    DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
    SCRIPT_PATH = os.path.join(BENCHMARK_DIR, "common", "download_data.py")
    DATA_FILE_LIST = ["2011_smd_hourly.xls", "2012_smd_hourly.xls",
                      "2013_smd_hourly.xls", "2014_smd_hourly.xls",
                      "2015_smd_hourly.xls", "2016_smd_hourly.xls",
                      "2017_smd_hourly.xlsx"]
    DATA_DIM_LIST = [[(57, 5), (8760, 16)] + [(8760, 14)]*8,
                     [(57, 5), (8784, 16)] + [(8784, 14)]*8,
                     [(59, 5), (8760, 16)] + [(8760, 14)]*8,
                     [(59, 5), (8760, 16)] + [(8760, 14)]*8 + [(0,1)],
                     [(57, 5), (8760, 16)] + [(8760, 14)]*8,
                     [(47, 10), (8784, 17)] + [(8784, 14)]*8,
                     [(51, 13), (8760, 21)] + [(8760, 14)]*8]
    # Remove data files if they are existed
    for f in DATA_FILE_LIST:
        file_path = os.path.join(DATA_DIR, f)
        if os.path.exists(file_path):
            os.remove(file_path)
        assert not os.path.exists(file_path)
    # Call data download script
    try:
        subprocess.check_output(["python", SCRIPT_PATH])
    except subprocess.CalledProcessError as e:
        print(e.output)
    # Check downloaded data (only check dimensions since download_data.py checks column names)
    for file_idx, f in enumerate(DATA_FILE_LIST):
        file_path = os.path.join(DATA_DIR, f)
        assert os.path.exists(file_path) 
        xls = pd.ExcelFile(file_path)
        for sheet_idx, s in enumerate(xls.sheet_names):
            assert xls.parse(s).shape == DATA_DIM_LIST[file_idx][sheet_idx]

