# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import pytest


from fclib.feature_engineering.feature_utils import (
    normalized_current_year,
    normalized_current_date,
    normalized_current_datehour,
    normalized_columns,
)

# normalization functions

sample_date = pd.to_datetime(pd.Series(["2000-01-01 12:30:59"]))


def test_normalized_current_year():
    dates = sample_date
    nyr = normalized_current_year(dates, 1980, 2020)
    assert all(nyr >= 0) and all(nyr <= 1)

    bad = normalized_current_year(dates, 2000, 2000)
    assert len(bad) == len(dates)


def test_normalized_current_date():
    dates = sample_date
    span = pd.to_datetime(pd.Series(["1980-01-01 00:00:00", "2020-01-01 23:59:59"]))
    ndt = normalized_current_date(dates, span[0], span[1])
    assert all(ndt >= 0) and all(ndt <= 1)

    badspan = pd.to_datetime(pd.Series(["2000-01-01 00:00:00", "2000-01-01 00:00:00"]))
    bad = normalized_current_date(dates, badspan[0], badspan[1])
    assert len(bad) == len(dates)


def test_normalized_current_datehour():
    dates = sample_date
    span = pd.to_datetime(pd.Series(["1980-01-01 00:00:00", "2020-01-01 23:59:59"]))
    ndt = normalized_current_datehour(dates, span[0], span[1])
    assert all(ndt >= 0) and all(ndt <= 1)

    badspan = pd.to_datetime(pd.Series(["2000-01-01 00:00:00", "2000-01-01 00:00:00"]))
    bad = normalized_current_datehour(dates, badspan[0], badspan[1])
    assert len(bad) == len(dates)


def test_normalized_columns():
    dates = pd.to_datetime(pd.Series(["2000-01-01", "2000-01-02", "2000-01-03"]))
    vals = pd.Series([1, 2, 3])

    nc1 = normalized_columns(dates, vals, mode="log")
    assert type(nc1).__name__ == "DataFrame"
    assert nc1.columns[0] == "normalized_columns"

    nc2 = normalized_columns(dates, vals, mode="minmax")
    assert all(nc2["normalized_columns"] >= 0) and all(nc2["normalized_columns"] <= 1)

    with pytest.raises(Exception):
        normalized_columns(dates, vals, mode="foo")
