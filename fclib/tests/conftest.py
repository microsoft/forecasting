# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd
import numpy as np
import pytest
from itertools import product


class GenerateData:
    @staticmethod
    def ojdata(start=50, stop=61):
        keyvars = {"store": [1, 2], "brand": [1, 2, 3], "week": list(range(start, stop))}
        df = pd.DataFrame([row for row in product(*keyvars.values())], columns=keyvars.keys())

        n = len(df)
        np.random.seed(12345)
        df["constant"] = 1
        df["logmove"] = np.random.normal(9, 1, n)
        df["price1"] = np.random.normal(0.55, 0.003, n)
        df["price2"] = np.random.normal(0.55, 0.003, n)
        df["price3"] = np.random.normal(0.55, 0.003, n)
        df["price4"] = np.random.normal(0.55, 0.003, n)
        df["price5"] = np.random.normal(0.55, 0.003, n)
        df["price6"] = np.random.normal(0.55, 0.003, n)
        df["price7"] = np.random.normal(0.55, 0.003, n)
        df["price8"] = np.random.normal(0.55, 0.003, n)
        df["price9"] = np.random.normal(0.55, 0.003, n)
        df["price10"] = np.random.normal(0.55, 0.003, n)
        df["price11"] = np.random.normal(0.55, 0.003, n)
        df["deal"] = np.random.binomial(1, 0.5, n)
        df["feat"] = np.random.binomial(1, 0.25, n)
        df["profit"] = np.random.normal(30, 7.5, n)
        return df


@pytest.fixture(scope="session")
def generate_ojdata():

    # data file that will be created and deleted each time test is run
    ojdata_csv = "fclib/tests/resources/ojdatagen.csv"
    df = GenerateData.ojdata()
    df.to_csv(ojdata_csv, index_label=False, index=False)

    yield generate_ojdata

    # teardown code
    try:
        os.remove(ojdata_csv)
        os.remove(os.path.dirname(ojdata_csv) + "/yx.csv")
    except Exception:
        pass


@pytest.fixture
def generate_data():
    return GenerateData
