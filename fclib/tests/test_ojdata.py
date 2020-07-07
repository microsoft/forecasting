# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import shutil
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory

from fclib.dataset.ojdata import download_ojdata, complete_and_fill_df, _gen_split_indices, split_train_test


# data file that will be created and deleted each time test is run
ojdata_csv = "fclib/tests/resources/ojdatagen.csv"


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


def test_complete_and_fill_df(generate_ojdata):
    ojdata = pd.read_csv(ojdata_csv, index_col=False)

    base_out = complete_and_fill_df(ojdata, stores=[1, 2], brands=[1, 2, 3], weeks=list(range(50, 61)))
    assert ojdata.equals(base_out)

    # remove cells
    cell_na = ojdata
    cell_na.loc[3:5, "logmove"] = np.nan
    cell_out = complete_and_fill_df(cell_na, stores=[1, 2], brands=[1, 2, 3], weeks=list(range(50, 61)))
    assert not any(pd.isna(cell_out["logmove"]))

    # remove rows
    row_na = ojdata
    row_na.drop(3, axis=0)
    row_out = complete_and_fill_df(row_na, stores=[1, 2], brands=[1, 2, 3], weeks=list(range(50, 61)))
    assert len(row_out) == len(ojdata)


def test_gen_split_indices(generate_ojdata):
    base = _gen_split_indices()
    assert len(base) == 3
    assert all([len(i) == 12 for i in base])

    small = _gen_split_indices(3, 2, 0, 50, 60)
    assert all([len(i) == 3 for i in small])


def test_split_train_test(generate_ojdata):
    resdir = os.path.dirname(ojdata_csv)
    shutil.copyfile(ojdata_csv, resdir + "/yx.csv")

    (traindf, testdf, auxdf) = split_train_test(resdir, 1, 2, 1, 50, 60)
    assert len(traindf) == 1
    assert len(testdf) == 1
    assert len(auxdf) == 1

    (traindf, testdf, auxdf) = split_train_test(resdir, 3, 2, 1, 50, 60)
    assert len(traindf) == 3
    assert len(testdf) == 3
    assert len(auxdf) == 3

    for i in list(range(3)):
        assert max(traindf[i].week) < min(testdf[i].week)
