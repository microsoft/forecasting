# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import datetime
import pytest


from fclib.feature_engineering.feature_utils import (
    df_from_cartesian_product,
    lagged_features,
    moving_averages,
    combine_features,
    gen_sequence_array,
    static_feature_array,
    normalize_columns,
    get_datetime_col,
    get_month_day_range,
    add_datetime,
)

# misc utilities


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
