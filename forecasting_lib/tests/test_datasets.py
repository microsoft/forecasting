import os
import subprocess
import pandas as pd


def test_download_retail_data():
    RETAIL_DIR = os.path.join(".", "forecasting_lib", "forecasting_lib", "dataset", "retail")
    DATA_DIR = os.path.join(".", "contrib", "tsperf", "OrangeJuice_Pt_3Weeks_Weekly", "data")
    SCRIPT_PATH = os.path.join(RETAIL_DIR, "download_data.r")
    DATA_FILE_LIST = ["yx.csv", "storedemo.csv"]
    # Remove data files if they are existed
    for f in DATA_FILE_LIST:
        file_path = os.path.join(DATA_DIR, f)
        if os.path.exists(file_path):
            os.remove(file_path)
        assert not os.path.exists(file_path)
    # Call data download script
    try:
        subprocess.call(["sudo", "Rscript", SCRIPT_PATH])
    except subprocess.CalledProcessError as e:
        print(e.output)
    # Check downloaded data
    DATA_DIM_LIST = [(106139, 19), (83, 12)]
    COLUMN_NAME_LIST = [
        [
            "store",
            "brand",
            "week",
            "logmove",
            "constant",
            "price1",
            "price2",
            "price3",
            "price4",
            "price5",
            "price6",
            "price7",
            "price8",
            "price9",
            "price10",
            "price11",
            "deal",
            "feat",
            "profit",
        ],
        [
            "STORE",
            "AGE60",
            "EDUC",
            "ETHNIC",
            "INCOME",
            "HHLARGE",
            "WORKWOM",
            "HVAL150",
            "SSTRDIST",
            "SSTRVOL",
            "CPDIST5",
            "CPWVOL5",
        ],
    ]
    for idx, f in enumerate(DATA_FILE_LIST):
        file_path = os.path.join(DATA_DIR, f)
        assert os.path.exists(file_path)
        df = pd.read_csv(file_path, index_col=None)
        assert df.shape == DATA_DIM_LIST[idx]
        assert list(df) == COLUMN_NAME_LIST[idx]
