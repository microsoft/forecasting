# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import datetime


from fclib.feature_engineering.feature_utils import (
    is_datetime_like,
    day_type,
    hour_of_day,
    time_of_year,
    week_of_year,
    week_of_month,
    month_of_year,
    day_of_week,
    day_of_month,
    day_of_year,
)


def test_is_datetime_like():
    st = "2000-01-01"
    assert not is_datetime_like(st)

    dt = datetime.datetime.now()
    assert is_datetime_like(dt)

    pdt = pd.DatetimeIndex(["2000-01-01"])
    assert is_datetime_like(pdt)

    pts = pd.Timestamp("2000-01-01T12:00:00")
    assert is_datetime_like(pts)

    d = datetime.date(2000, 1, 1)
    assert is_datetime_like(d)


def test_day_type():
    dates = pd.to_datetime(pd.Series(["2000-01-01", "2000-01-02", "2000-01-03"]))
    hols = pd.Series([True, False, False])

    dty = day_type(dates)
    assert all(dty == [5, 6, 0])

    dty2 = day_type(dates, hols)
    assert all(dty2 == [7, 8, 0])


# date component extractors

sample_date = pd.to_datetime(pd.Series(["2000-01-01 12:30:59"]))


def test_hour_of_day():
    dates = sample_date
    assert all(hour_of_day(dates) == 12)


def test_time_of_year():
    dates = sample_date
    tyr = time_of_year(dates)
    assert all(tyr >= 0 and tyr <= 1)


def test_week_of_year():
    dates = sample_date
    assert week_of_year(dates)[0] == 52  # first day of 2000 is in last week of 1999


def test_week_of_month():
    dates = sample_date
    assert week_of_month(dates)[0] == 1  # first day of 2000 is in first month of 2000


def test_month_of_year():
    dates = sample_date
    assert month_of_year(dates)[0] == 1


def test_day_of_week():
    dates = sample_date
    assert day_of_week(dates)[0] == 5


def test_day_of_month():
    dates = sample_date
    assert day_of_month(dates)[0] == 1


def test_day_of_year():
    dates = sample_date
    assert day_of_year(dates)[0] == 1


# def test_encoded_month_of_year():
#     dates = sample_date
#     enc = encoded_month_of_year(dates)
#     assert len(enc.columns) == 12

# def test_encoded_day_of_week():
#     dates = sample_date
#     enc = encoded_day_of_week(dates)
#     assert len(enc.columns) == 7

# def test_encoded_day_of_year():
#     dates = sample_date
#     enc = encoded_day_of_year(dates)
#     assert len(enc.columns) >= 365

# def test_encoded_hour_of_day():
#     dates = sample_date
#     enc = encoded_hour_of_day(dates)
#     assert len(enc.columns) == 24

# def test_encoded_week_of_year():
#     dates = sample_date
#     enc = encoded_week_of_year(dates)
#     assert len(enc.columns) == 53
