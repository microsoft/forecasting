from datetime import timedelta
import calendar
import pandas as pd
import numpy as np
import warnings
from math import ceil

from abc import ABC, abstractmethod
from tsperf.feature_engineering.base_ts_estimators import BaseTSFeaturizer


class TemporalFeaturizer(BaseTSFeaturizer):
    """
    Computes commonly used time-related features.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.

        feature_list(list of str, optional): list of temporal features to
            return. The following features are available:
                hour_of_day
                week_of_year
                month_of_year
                day_of_week
                day_of_month
                day_of_year
                normalized_hour_of_year
                week_of_year
                year

            If feature_list is not specified, a default set of features are
            returned depending on df_config['frequency']
            For 'hourly' data: hour_of_day, day_of_week, week_of_year,
            month_of_year
            For 'daily' data: day_of_week, week_of_year, month_of_year
            For 'weekly' data: week_of_year, month_of_year
            For 'monthly' data: month_of_year
    """

    def __init__(self, df_config, feature_list=None):
        super().__init__(df_config)

        if feature_list:
            self.feature_list = feature_list
        elif self.frequency == "H":
            self.feature_list = [
                "hour_of_day",
                "day_of_week",
                "week_of_year",
                "month_of_year",
            ]
        elif self.frequency == "D":
            self.feature_list = [
                "day_of_week",
                "week_of_year",
                "month_of_year",
            ]
        elif self.frequency == "W":
            self.feature_list = ["week_of_year", "month_of_year"]
        elif self.frequency == "M":
            self.feature_list = ["month_of_year"]
        else:
            raise Exception(
                "Please specify the feature_list, because the "
                "data frequency is not H, D, W, or M"
            )

        self._feature_function_map = {
            "hour_of_day": self.hour_of_day,
            "week_of_year": self.week_of_year,
            "month_of_year": self.month_of_year,
            "day_of_week": self.day_of_week,
            "day_of_month": self.day_of_month,
            "day_of_year": self.day_of_year,
            "normalized_hour_of_year": self.normalized_hour_of_year,
            "week_of_month": self.week_of_month,
            "year": self.year,
        }

    @staticmethod
    def hour_of_day(time_col):
        """Returns the hour from a datetime column."""
        return time_col.dt.hour

    @staticmethod
    def week_of_year(time_col):
        """Returns the week from a datetime column."""
        return time_col.dt.week

    @staticmethod
    def month_of_year(time_col):
        """Returns the month from a datetime column."""
        return time_col.dt.month

    @staticmethod
    def day_of_week(time_col):
        """Returns the day of week from a datetime column."""
        return time_col.dt.dayofweek

    @staticmethod
    def day_of_month(time_col):
        """Returns the day of month from a datetime column."""
        return time_col.dt.day

    @staticmethod
    def day_of_year(time_col):
        """Returns the day of year from a datetime column."""
        return time_col.dt.dayofyear

    @staticmethod
    def week_of_month(time_col):
        """Returns the week of month from a datetime column."""
        first_day = time_col.apply(lambda x: x.replace(day=1))
        dom = time_col.dt.day
        adjusted_dom = dom + first_day.dt.dayofweek
        wom = adjusted_dom.apply(lambda x: int(ceil(x / 7.0)))
        return wom

    @staticmethod
    def year(time_col):
        """Returns year from a datetime column."""
        return time_col.dt.year

    @staticmethod
    def normalized_hour_of_year(time_col):
        """
        Normalized hour of year is a cyclic variable that indicates the annual
        position of a particular hour on a particular day and repeats each
        year. It is each year linearly increasing over time going from 0 on
        January 1 at 00:00 to 1 on December 31st at 23:00. The values
        are normalized to be between [0, 1].

        Args:
            time_col: Datetime column.

        Returns:
            A numpy array containing converted the time column to hour of year.
        """

        time_of_year = pd.DataFrame(
            {
                "day_of_year": time_col.dt.dayofyear,
                "hour_of_day": time_col.dt.hour,
                "year": time_col.dt.year,
            }
        )
        time_of_year["normalized_hour_of_year"] = (
            time_of_year["day_of_year"] - 1
        ) * 24 + time_of_year["hour_of_day"]

        time_of_year["year_length"] = time_of_year["year"].apply(
            lambda y: 366 if calendar.isleap(y) else 365
        )

        time_of_year["normalized_hour_of_year"] = time_of_year[
            "normalized_hour_of_year"
        ] / (time_of_year["year_length"] * 24 - 1)

        return time_of_year["normalized_hour_of_year"].values

    def fit(self, X, y=None):
        """
        To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer
        """
        return self

    def transform(self, X):
        """
        Creates time-related features from the time column of the input data.
        Args:
            X(pandas.DataFrame): Input data frame to create features on.
        Returns:
            pandas.DataFrame: Output data frame with time features added to the
                input data frame
        """
        self._check_config_cols_exist(X)
        X = X.copy()
        time_col = self._get_time_col(X)
        for feature in self.feature_list:
            if feature in X.columns:
                warnings.warn(
                    "Column {} is already in the data frame, "
                    "it will be overwritten.".format(feature)
                )
            feature_function = self._feature_function_map[feature]
            X[feature] = feature_function(time_col)

        return X


class DayTypeFeaturizer(BaseTSFeaturizer):

    """
    Convert the time column of the input data frame to 7 day types.
    The following mapping is used to convert day of week or holiday to
    integers.
        Monday: 0
        Tuesday, Wednesday, and Thursday: 2
        Friday:4
        Saturday: 5
        Sunday: 6
        Holiday: 7
        Days before and after a holiday: 8

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        holiday_col_name(str, optional): Name of the holiday column. The
            holiday column should contain integers greater than 0 representing
            holidays. The mapping between the holiday and the integer does
            not matter for this featurizer.
        semi_holiday_offset(datetime.timedelta, optional): The time range
            before and after each holiday to be considered as semi-holiday.
            Default value timedelta(days=1).
        weekday_type_map(dict, optional): Mapping multiple weekdays to the same
            number. By default, Tuesday (1) and Thursday (3) are mapped to 2,
            so that Tuesday, Wednesday, and Thursday are treated the same.
        holiday_code(int, optional): Integer used to represent holidays.
            Default value is 7.
        semi_holiday_code(int, optional): Integer used to represent days
            before and after  holiday. Default value is 8

    """

    # TODO: Update to use the Python holiday package to make this function
    # more generic
    def __init__(
        self,
        df_config,
        holiday_col_name=None,
        semi_holiday_offset=timedelta(days=1),
        weekday_type_map={1: 2, 3: 2},
        holiday_code=7,
        semi_holiday_code=8,
    ):
        super().__init__(df_config)
        self.weekday_type_map = weekday_type_map

        self.holiday_col_name = holiday_col_name
        self.semi_holiday_offset = semi_holiday_offset
        self.holiday_code = holiday_code
        self.semi_holiday_code = semi_holiday_code

    def fit(self, X, y=None):
        """
        To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer
        """
        return self

    def transform(self, X):
        """
        Creates a day type  feature from the time column of the input data.
        Args:
            X(pandas.DataFrame): Input data frame to create features on.
        Returns:
            pandas.DataFrame: Output data frame with a day_type feature
            column added to the input data frame
        """
        self._check_config_cols_exist(X)
        X = X.copy()
        time_col = self._get_time_col(X)

        datetype = pd.DataFrame({"day_type": time_col.dt.dayofweek})
        datetype.replace({"day_type": self.weekday_type_map}, inplace=True)

        if self.holiday_col_name is not None:
            holiday_col = X[self.holiday_col_name].values
            holiday_mask = holiday_col > 0
            datetype.loc[holiday_mask, "day_type"] = self.holiday_code

            # Create a temporary Date column to calculate dates near
            # the holidays
            datetype["Date"] = pd.to_datetime(
                time_col.dt.date, format=self.time_format
            )
            holiday_dates = set(datetype.loc[holiday_mask, "Date"])

            semi_holiday_dates = [
                pd.date_range(
                    start=d - self.semi_holiday_offset,
                    end=d + self.semi_holiday_offset,
                    freq="D",
                )
                for d in holiday_dates
            ]

            # Flatten the list of lists
            semi_holiday_dates = [
                d for dates in semi_holiday_dates for d in dates
            ]

            semi_holiday_dates = set(semi_holiday_dates)
            semi_holiday_dates = semi_holiday_dates.difference(holiday_dates)

            datetype.loc[
                datetype["Date"].isin(semi_holiday_dates), "day_type"
            ] = self.semi_holiday_code

        X["day_type"] = datetype["day_type"].values

        return X


class BaseFourierFeaturizer(BaseTSFeaturizer, ABC):
    """
    Base abstract class for Fourier featurizers.
    Child classes need to implement _get_time_values for a specific type of
    Fourier feature.
    """

    @property
    def period(self):
        return self._period

    @abstractmethod
    def _get_time_values(self, datetime_col):
        """
        Abstract method for computing the time values, e.g. hour of day,
        day of week, week of year, to compute Fourier approximation on.
        """
        pass

    @staticmethod
    def fourier_approximation(t, n, period):
        """
        Computes Fourier Series at different harmonies (n) and periods.

        Args:
            t(int): Time position, e.g. for daily Fourier series, this is the
                hour of day. For weekly Fourier, this is the day of week. For
                annual Fourier, this is the day of year.
            n(int): Harmony to compute, n = 1, 2, 3,...
            period(int): Period of the time series data. For hourly data with
                daily seasonality, this is 24. For weekly seasonality, this is
                7.
                For yearly seasonality, this is the number of days in a year.

        Returns:
            float: Sine component
            float: Cosine component
        """
        x = n * 2 * np.pi * t / period
        x_sin = np.sin(x)
        x_cos = np.cos(x)

        return x_sin, x_cos

    def fit(self, X, y=None):
        """
        To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this transformer
        """
        return self

    def transform(self, X):
        """
        Creates Fourier features from the time column of the input data.
        Args:
            X(pandas.DataFrame): Input data frame to create features on.
        Returns:
            pandas.DataFrame: Output data frame with Fourier feature
            columns added to the input data frame
        """
        self._check_config_cols_exist(X)
        X = X.copy()
        datetime_col = self._get_time_col(X)

        time_values = self._get_time_values(datetime_col)
        output_dict = {}
        for n in range(1, self.n_harmonics + 1):
            sin, cos = self.fourier_approximation(time_values, n, self.period)

            output_dict[self.output_prefix + "_sin_" + str(n)] = sin
            output_dict[self.output_prefix + "_cos_" + str(n)] = cos

        for k, v in output_dict.items():
            X[k] = v

        return X


class AnnualFourierFeaturizer(BaseFourierFeaturizer):
    """
    Creates Annual Fourier Series at different harmonies (n).

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        n_harmonics(int): Number of harmonies to compute, n=1, 2, 3,...
        feature_frequency(str): Frequency level to compute Fourier feature
            on. The following values are accepted
            day: Convert day of year to Fourier feature with a period of 365.24
            week: Convert week of year to Fourier feature with a period of 52.
            month: Convert month of year to Fourier feature with a period of
                12.
            quarter: Convert quarter of year to Fourier feature with a period
                of 4.
    """

    def __init__(self, df_config, n_harmonics, feature_frequency="day"):
        super().__init__(df_config)
        self.n_harmonics = n_harmonics
        self.output_prefix = "annual"
        self.feature_frequency = feature_frequency

    @property
    def feature_frequency(self):
        return self._feature_frequency

    @feature_frequency.setter
    def feature_frequency(self, val):
        if val == "day":
            self._period = 365.24
        elif val == "week":
            self._period = 52
        elif val == "month":
            self._period = 12
        elif val == "quarter":
            self._period = 4
        else:
            raise ValueError(
                "Invalid feature_frequency, {}, received. "
                "Accepted values are day, week, month, and "
                "quarter".format(val)
            )
        self._feature_frequency = val

    def _get_time_values(self, datetime_col):
        if self.feature_frequency == "day":
            return datetime_col.dt.dayofyear
        elif self.feature_frequency == "week":
            return datetime_col.dt.week
        elif self.feature_frequency == "month":
            return datetime_col.dt.month
        elif self.feature_frequency == "quarter":
            return datetime_col.dt.quarter


class WeeklyFourierFeaturizer(BaseFourierFeaturizer):
    """
    Creates Weekly Fourier Series at different harmonies (n).

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        n_harmonics(int): Number of harmonies to compute, n=1, 2, 3,...

    """

    def __init__(self, df_config, n_harmonics):
        super().__init__(df_config)
        self.n_harmonics = n_harmonics
        self.output_prefix = "weekly"
        self._period = 7

    @staticmethod
    def _get_time_values(datetime_col):
        return datetime_col.dt.dayofweek + 1


class DailyFourierFeaturizer(BaseFourierFeaturizer):
    """
    Creates Daily Fourier Series at different harmonies (n).

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        n_harmonics(int): Number of harmonies to compute, n=1, 2, 3,...
    """

    def __init__(self, df_config, n_harmonics):
        super().__init__(df_config)
        self.n_harmonics = n_harmonics
        self.output_prefix = "daily"
        self._period = 24

    @staticmethod
    def _get_time_values(datetime_col):
        return datetime_col.dt.hour + 1
