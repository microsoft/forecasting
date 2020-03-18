# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base classes for time series forecasting tasks.
"""
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from common.utils import (
    is_datetime_like,
    is_iterable_but_not_string,
    get_offset_by_frequency,
)


class BaseTSEstimator(BaseEstimator, ABC):
    """
    Base abstract class for all featurizer and model classes.
    Args:
        df_config(dict): A dictionary defining common columns in time
            series forecasting tasks.
            time_col_name(str): Name of the column with timestamps.
            target_col_name(str): Name of the target value column to
                forecast.
            ts_id_col_names(str or list of str): Name(s) of a column or a
                list of columns used to identify unique time series.
                For example, if a data frame contains data of multiple
                stores and multiple brands, each store and brand
                combination defines a unique time series.
            frequency(str): Frequency of the data frame.
                See https://pandas.pydata.org/pandas-docs/stable/user_guide/
                timeseries.html#dateoffset-objects
            time_format: Format of the timestamps in the time column.
                See http://strftime.org/.

    """

    def __init__(self, df_config):
        self.time_col_name = df_config["time_col_name"]
        self.target_col_name = df_config["target_col_name"]
        self.ts_id_col_names = df_config["ts_id_col_names"]
        self.frequency = df_config["frequency"]
        self.time_format = df_config["time_format"]

        self._offset = get_offset_by_frequency(self.frequency)

    @property
    def ts_id_col_names(self):
        return self._ts_id_col_names

    @ts_id_col_names.setter
    def ts_id_col_names(self, val):
        # If ts_id_col_names is not a list, convert it to a list to simplify
        # downstream code that use it.
        if val is None:
            self._ts_id_col_names = []
        elif is_iterable_but_not_string(val):
            self._ts_id_col_names = list(val)
        else:
            self._ts_id_col_names = [val]

    def _check_config_cols_exist(self, df):
        """
        Checks if the columns specified in data frame configuration exist.
        """
        if self.time_col_name not in df.columns:
            raise Exception("time_col_name {} does not exist in the input " "data frame".format(self.time_col_name))
        for id_col_name in self.ts_id_col_names:
            if id_col_name not in df.columns:
                raise Exception("ts_id_col_names {} does not exist in the " "input data frame".format(id_col_name))

    def _get_time_col(self, df):
        """
        Return the time column of the input data frame in pandas.datetime type.
        """
        time_col = df[self.time_col_name]
        if not is_datetime_like(time_col):
            time_col = pd.to_datetime(time_col, format=self.time_format)
        return time_col


class BaseTSFeaturizer(BaseTSEstimator, TransformerMixin):
    """
    Base abstract featurizer class for all time series featurizers.
    """

    def fit(self, X, y=None):
        """
        Default implementation of fit method. By default, nothing is done
        in the fit method.
        """
        return self

    @abstractmethod
    def transform(self, X):
        """
        Abstract transform method. Child classes must implement transform
        method that create features on the input data frame.
        Args:
            X(pandas.DataFrame): Input data frame to create features on.
        Returns:
            pandas.DataFrame: Output data frame with features added to the
                input data frame

        """
        return X
