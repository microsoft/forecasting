# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import datetime


from fclib.feature_engineering.feature_utils import annual_fourier, weekly_fourier, daily_fourier, fourier_approximation

# Fourier stuff


def test_fourier_approximation():
    dates = pd.Series([x for x in range(1, 366)])
    (fsin, fcos) = fourier_approximation(dates, 1, 365.24)
    assert len(fsin) == len(dates)
    assert len(fcos) == len(dates)
    assert all(abs(fsin) <= 1) and all(abs(fcos) <= 1)


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
