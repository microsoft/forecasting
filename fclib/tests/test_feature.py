# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import datetime
import pytest


from fclib.feature_engineering.feature_utils import *

def test_is_datetime_like():
    st = '2000-01-01'
    assert (not is_datetime_like(st))

    dt = datetime.datetime.now()
    assert is_datetime_like(dt)

    pdt = pd.DatetimeIndex(['2000-01-01'])
    assert is_datetime_like(pdt)

    pts = pd.Timestamp('2000-01-01T12:00:00')
    assert is_datetime_like(pts)

    d = datetime.date(2000, 1, 1)
    assert is_datetime_like(d)


def test_day_type():
    dates = pd.to_datetime(pd.Series(['2000-01-01', '2000-01-02', '2000-01-03']))
    hols = pd.Series([1])

    dty = day_type(dates)
    assert all(dty == [5, 6, 0])

    dty2 = day_type(dates, hols)
    assert all(dty2 == [7, 8, 0])

# date component extractors
def test_hour_of_day():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    assert all(hour_of_day(dates) == 12)

def test_time_of_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    tyr = time_of_year(dates)
    assert all(tyr >= 0 and tyr <= 1)

def test_week_of_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    assert week_of_year(dates)[0] == 52  # first day of 2000 is in last week of 1999

def test_week_of_month():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    assert week_of_month(dates)[0] == 1  # first day of 2000 is in first month of 2000

def test_month_of_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    assert month_of_year(dates)[0] == 1

def test_day_of_week():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    assert day_of_week(dates)[0] == 5

def test_day_of_month():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    assert day_of_month(dates)[0] == 1

def test_day_of_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    assert day_of_year(dates)[0] == 1

def test_encoded_month_of_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    enc = encoded_month_of_year(dates)
    assert len(enc.columns) == 12

def test_encoded_day_of_week():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    enc = encoded_day_of_week(dates)
    assert len(enc.columns) == 7

def test_encoded_day_of_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    enc = encoded_day_of_year(dates)
    assert len(enc.columns) >= 365

def test_encoded_hour_of_day():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    enc = encoded_hour_of_day(dates)
    assert len(enc.columns) == 24

def test_encoded_week_of_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    enc = encoded_week_of_year(dates)
    assert len(enc.columns) >= 52

# normalization functions
def test_normalized_current_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    nyr = normalized_current_year(dates, 1980, 2020)
    assert all(nyr >= 0 and nyr <= 1)

def test_normalized_current_date():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    span = pd.to_datetime(pd.Series(['1980-01-01 00:00:00', '2020-01-01 23:59:59']))
    ndt = normalized_current_date(dates, span[0], span[1])
    assert all(ndt >= 0 and ndt <= 1)

def test_normalized_current_datehour():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    span = pd.to_datetime(pd.Series(['1980-01-01 00:00:00', '2020-01-01 23:59:59']))
    ndt = normalized_current_datehour(dates, span[0], span[1])
    assert all(ndt >= 0 and ndt <= 1)

def test_normalized_columns():
    dates = pd.to_datetime(pd.Series(['2000-01-01', '2000-01-02', '2000-01-03']))
    vals = pd.Series([1, 2, 3])

    nc1 = normalized_columns(dates, vals, mode="log")
    assert type(nc1).__name__ == "DataFrame"
    assert nc1.columns[0] == "normalized_columns"

    nc2 = normalized_columns(dates, vals, mode="minmax")
    assert all(nc2["normalized_columns"] >= 0) and all(nc2["normalized_columns"] <= 1)

    with pytest.raises(Exception):
        normalized_columns(dates, vals, mode="foo")
