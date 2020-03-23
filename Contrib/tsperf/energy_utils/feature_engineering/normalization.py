# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
A set of classes  for creating normalized time features. These types of
features are useful for capturing trend in time series data.
"""
from abc import abstractmethod
import warnings
import pandas as pd
from fclib.feature_engineering.utils import is_datetime_like
from fclib.feature_engineering.base_ts_estimators import BaseTSFeaturizer


class BaseTemporalNormalizer(BaseTSFeaturizer):
    """
    Base abstract class for creating normalized time features.
    Child class should implement _get_min_max_time and _normalize_time to
    create normalized features at a specific granularity, e.g. normalized
    year, normalized day.
    """

    @abstractmethod
    def _get_min_max_time(self, time_col):
        """
        Returns (min_time, max_time) used by _normalize_time to normalize
        time.
        """
        pass

    @abstractmethod
    def _normalize_time(self, time_col):
        """Returns normalized time_col."""
        pass

    def fit(self, X, y=None):
        """
        Computes and returns the minimum and maximum time on the training data.

        Args:
             X(pandas.DataFrame): Input data frame to fit on.

        Returns:
            TemporalNormalizer: The fitted temporal normalizer.
        """
        time_col = self._get_time_col(X)
        self.min_time, self.max_time = self._get_min_max_time(time_col)

        self._is_fit = True

        return self

    def transform(self, X):
        """
        Creates a new column by normalizing the time column of the input data
        at certain granularity, e.g. day, hour, year, etc.
        Args:
            X(pandas.DataFrame): Input data frame with the time column to
                normalize.
        Returns:
            pandas.DataFrame: Output data frame with the normalized time
            added to the input data frame.
        """
        if not self._is_fit:
            raise Exception("The featurizer needs to be fitted first to be " "used to transform data.")
        self._check_config_cols_exist(X)
        if self._output_col_name in X.columns:
            warnings.warn(
                "Column {} is already in the data frame, " "it will be overwritten.".format(self._output_col_name)
            )

        time_col = X[self.time_col_name]
        if not is_datetime_like(time_col):
            time_col = pd.to_datetime(time_col, format=self.time_format)

        X[self._output_col_name] = self._normalize_time(time_col)

        return X


class YearNormalizer(BaseTemporalNormalizer):
    """
    Creates a temporal feature indicating the position of the year of a record
    in the entire time period under consideration, normalized to be between
    0 and 1 on training data and greater than 1 on testing data.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.

    """

    def __init__(self, df_config):
        super(YearNormalizer, self).__init__(df_config)
        self._is_fit = False
        self._output_col_name = "normalized_year"

    @classmethod
    def _get_min_max_time(cls, time_col):
        min_year = time_col.dt.year.min()
        max_year = time_col.dt.year.max()

        return min_year, max_year

    def _normalize_time(self, time_col):
        year = time_col.dt.year
        if self.max_time != self.min_time:
            normalized_time = (year - self.min_time) / (self.max_time - self.min_time)
        else:
            normalized_time = 0

        return normalized_time


class DateNormalizer(BaseTemporalNormalizer):
    """
    Creates a temporal feature indicating the position of the date of a record
    in the entire time period under consideration, normalized to be between
    0 and 1 on training data and greater than 1 on testing data.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
    """

    def __init__(self, df_config):
        super(DateNormalizer, self).__init__(df_config)
        self._is_fit = False
        self._output_col_name = "normalized_date"

    @classmethod
    def _get_min_max_time(cls, time_col):
        min_date = time_col.dt.date.min()
        max_date = time_col.dt.date.max()

        return min_date, max_date

    def _normalize_time(self, time_col):
        date = time_col.dt.date
        current_date = (date - self.min_time).apply(lambda x: x.days)

        if self.max_time != self.min_time:
            normalized_time = current_date / (self.max_time - self.min_time).days
        else:
            normalized_time = 0

        return normalized_time


class DateHourNormalizer(BaseTemporalNormalizer):
    """
    Creates a temporal feature indicating the position of the hour of a record
    in the entire time period under consideration, normalized to be between
    0 and 1 on training data and greater than 1 on testing data.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
    """

    def __init__(self, df_config):
        super(DateHourNormalizer, self).__init__(df_config)
        self._is_fit = False
        self._output_col_name = "normalized_datehour"

    @classmethod
    def _get_min_max_time(cls, time_col):
        min_datehour = time_col.min()
        max_datehour = time_col.max()

        return min_datehour, max_datehour

    def _normalize_time(self, time_col):
        current_datehour = (time_col - self.min_time).apply(lambda x: x.days * 24 + x.seconds / 3600)

        max_min_diff = self.max_time - self.min_time

        if max_min_diff != 0:
            normalized_time = current_datehour / (max_min_diff.days * 24 + max_min_diff.seconds / 3600)
        else:
            normalized_time = 0

        return normalized_time
