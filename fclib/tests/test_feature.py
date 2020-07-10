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

# def test_encoded_month_of_year():
#     dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
#     enc = encoded_month_of_year(dates)
#     assert len(enc.columns) == 12

# def test_encoded_day_of_week():
#     dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
#     enc = encoded_day_of_week(dates)
#     assert len(enc.columns) == 7

# def test_encoded_day_of_year():
#     dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
#     enc = encoded_day_of_year(dates)
#     assert len(enc.columns) >= 365

# def test_encoded_hour_of_day():
#     dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
#     enc = encoded_hour_of_day(dates)
#     assert len(enc.columns) == 24

# def test_encoded_week_of_year():
#     dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
#     enc = encoded_week_of_year(dates)
#     assert len(enc.columns) >= 52

# normalization functions

def test_normalized_current_year():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    nyr = normalized_current_year(dates, 1980, 2020)
    assert all(nyr >= 0) and all(nyr <= 1)

def test_normalized_current_date():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    span = pd.to_datetime(pd.Series(['1980-01-01 00:00:00', '2020-01-01 23:59:59']))
    ndt = normalized_current_date(dates, span[0], span[1])
    assert all(ndt >= 0) and all(ndt <= 1)

def test_normalized_current_datehour():
    dates = pd.to_datetime(pd.Series(['2000-01-01 12:30:59']))
    span = pd.to_datetime(pd.Series(['1980-01-01 00:00:00', '2020-01-01 23:59:59']))
    ndt = normalized_current_datehour(dates, span[0], span[1])
    assert all(ndt >= 0) and all(ndt <= 1)

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

# Fourier stuff

def test_fourier_approximation():
    dates = pd.Series([x for x in range(1, 366)])
    (fsin, fcos) = fourier_approximation(dates, 1, 365.24)
    assert len(fsin) == len(dates)
    assert len(fcos) == len(dates)

def test_annual_fourier():
    dates = pd.to_datetime(pd.Series([datetime.date(2000, 1, 1) + datetime.timedelta(days=x) for x in range(365)]))
    fa = annual_fourier(dates, 5)
    assert len(fa) == 10

def test_weekly_fourier():
    dates = pd.to_datetime(pd.Series([datetime.date(2000, 1, 1) + datetime.timedelta(days=x) for x in range(365)]))
    fw = weekly_fourier(dates, 5)
    assert len(fw) == 10

def test_daily_fourier():
    dates = pd.to_datetime(pd.Series([datetime.date(2000, 1, 1) + datetime.timedelta(days=x) for x in range(365)]))
    fd = daily_fourier(dates, 5)
    assert len(fd) == 10
    

# other

def test_df_from_cartesian_product():
    d = {"x1": [1, 2, 3], "x2": [4, 5, 6], "x3": ["a", "b", "c"]}
    df = df_from_cartesian_product(d)
    assert len(df) == 27
    assert list(df.columns) == ["x1", "x2", "x3"]

def test_lagged_features():
    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6], "x3": ["a", "b", "c"]})
    dflag = lagged_features(df, [1, 2])
    assert dflag.shape == (3, 6)
    assert all(pd.isna(dflag.iloc[0, :]))

def test_moving_averages():
    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    dfma = moving_averages(df, 1, 2)
    assert dfma.shape == (3, 2)
    assert all(pd.isna(dfma.iloc[0, :]))

def test_combine_features():
    df = pd.DataFrame({"x1": [1, 2, 3], "x2": [4, 5, 6]})
    dfcomb = combine_features(df, ["x1", "x2"], [1, 2], 2, ["x1", "x2"])
    assert dfcomb.shape == (3, 8)

def test_gen_sequence_array():
    val = pd.Series(x for x in range(8))
    df0 = df_from_cartesian_product({"x1": [1, 2], "x2": [1, 2, 3, 4]})
    df = pd.concat([val.to_frame("y"), df0], axis=1)
    arr = gen_sequence_array(df, 2, ["y"], "x1", "x2")
    assert len(arr) == 8

def test_static_feature_array():
    val = pd.Series(x for x in range(8))
    df0 = df_from_cartesian_product({"x1": [1, 2], "x2": [1, 2, 3, 4]})
    df = pd.concat([val.to_frame("y"), df0], axis=1)
    arr = static_feature_array(df, 8, ["x1", "x2"], "x1", "x2")
    assert len(arr) == 8

def test_normalize_columns():
    df = pd.Series((x * 1.0) for x in range(20)).to_frame("x")
    (sc, _) = normalize_columns(df, ["x"])
    assert len(sc) == len(df)
    assert all(sc["x"] >= 0) and all(sc["x"] <= 1)

def test_get_datetime_col():
    df = pd.DataFrame({"x1": ["2001-01-01", "2001-01-02", "2001-01-03"], "x2": [1, 2, 3], "x3": ["a", "b", "c"]})
    dt1 = get_datetime_col(df, "x1")
    assert len(dt1) == 3

    with pytest.raises(Exception):
        get_datetime_col(df, "x3")

def test_get_month_day_range():
    x = datetime.datetime(2000, 1, 15)
    (first, last) = get_month_day_range(x)
    assert first == datetime.datetime(2000, 1, 1, 0, 0)
    assert last == datetime.datetime(2000, 1, 31, 23, 0)

def test_add_datetime():
    x = datetime.datetime(2000, 1, 1)
    xy = add_datetime(x, "year", 1)
    assert xy == datetime.datetime(2001, 1, 1)

    xm = add_datetime(x, "month", 1)
    assert xm == datetime.datetime(2000, 2, 1)

    xd = add_datetime(x, "day", 1)
    assert xd == datetime.datetime(2000, 1, 2)
