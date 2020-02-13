# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd
from tempfile import TemporaryDirectory

from forecasting_lib.dataset.ojdata import download_ojdata


def test_download_retail_data():

    DATA_FILE_LIST = ["yx.csv", "storedemo.csv"]

    with TemporaryDirectory() as tmpdirname:
        print("Created temporary directory", tmpdirname)

        # Download the data to the temp directory
        download_ojdata(tmpdirname)
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
            file_path = os.path.join(tmpdirname, f)
            assert os.path.exists(file_path)
            df = pd.read_csv(file_path, index_col=None)
            assert df.shape == DATA_DIM_LIST[idx]
            assert list(df) == COLUMN_NAME_LIST[idx]
