# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from abc import abstractmethod
import numpy as np
import warnings

from forecasting_lib.feature_engineering.base_ts_estimators import BaseTSFeaturizer
from forecasting_lib.feature_engineering.utils import convert_to_tsdf, is_iterable_but_not_string


class BaseRollingWindowFeaturizer(BaseTSFeaturizer):
    """
    Base abstract class for rolling window features.
    Child classes need to implement the method _rolling_window_agg_single_ts,
    which creates rolling window features on a data frame containing a single
    time series.
    """

    @property
    def input_col_names(self):
        return self._input_col_names

    @input_col_names.setter
    def input_col_names(self, val):
        if is_iterable_but_not_string(val):
            self._input_col_names = list(val)
        else:
            self._input_col_names = [val]

    # future_value_available is a read-only property because there are a few
    # other properties depend on it.
    @property
    def future_value_available(self):
        return self._future_value_available

    @property
    def max_horizon(self):
        return self._max_horizon

    @max_horizon.setter
    def max_horizon(self, val):
        if not val and not self.future_value_available:
            raise Exception("max_horizon must be set when " "future_value_available is False")
        self._max_horizon = val

    @property
    def window_args(self):
        return self._window_args

    @window_args.setter
    def window_args(self, val):
        # If future value is not available, force to set the labels at the
        # right end of the window to avoid data leakage.
        if not self.future_value_available and "center" in val and val["center"] is True:
            val["center"] = False
            warnings("window_args['center'] is set to False, because " "future_value_available is False")
        self._window_args = val

    @abstractmethod
    def _rolling_window_agg_single_ts(self, input_df, forecast_creation_time):
        """
        Creates rolling window features on a single time series data frame.

        Args:
            input_df(pandas.DataFrame): Input data frame to create rolling
                window features on.
            forecast_creation_time(pandas.datetime): A timestamp specifying
                when the rolling window features are created.
        Returns:
            pandas.DataFrame: Data frame with the time column of input_df
                and rolling window features.
        """
        pass

    def fit(self, X, y=None):
        """
        To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this featurizer.
        """
        return self

    def transform(self, X):
        """
        Creates rolling window features on a subset of columns in X.

        Args:
             X(pandas.DataFrame): Input data frame to create rolling window
                features on.
        Returns:
            pandas.DataFrame: Output data frame with the rolling window
                features added to the input data frame.
        """
        self._check_config_cols_exist(X)

        col_names = [self.time_col_name] + self.input_col_names + self.ts_id_col_names
        merge_col_names = [self.time_col_name] + self.ts_id_col_names

        if self.train_df is not None:
            self._check_config_cols_exist(self.train_df)
            time_col = self._get_time_col(self.train_df)
            forecast_creation_time = time_col.max()
            X_tmp = pd.concat([self.train_df, X], sort=True)
            X_tmp = X_tmp[col_names].copy()
        else:
            time_col = self._get_time_col(X)
            if not self.future_value_available:
                # Compute an imaginary forecast creation time for the training
                # data based on the maximum timestamp to forecast on
                max_train_timestamp = time_col.max()
                forecast_creation_time = max_train_timestamp - self.max_horizon * self._offset
            else:
                forecast_creation_time = time_col.max()
            X_tmp = X[col_names].copy()

        if len(self.ts_id_col_names) == 0:
            output_tmp = self._rolling_window_agg_single_ts(X_tmp, forecast_creation_time)
        else:
            output_tmp = X_tmp.groupby(self.ts_id_col_names).apply(
                lambda g: self._rolling_window_agg_single_ts(g, forecast_creation_time)
            )
            output_tmp.reset_index(inplace=True)

        if self.train_df is not None:
            output_tmp = output_tmp.loc[output_tmp[self.time_col_name] > forecast_creation_time].copy()

        X = pd.merge(X, output_tmp, on=merge_col_names)
        if X.shape[0] == 0:
            raise Exception(
                "The featurizer output is empty. Set the "
                "train_df property of the featurizer to "
                "None if transforming training data."
            )
        return X


class RollingWindowFeaturizer(BaseRollingWindowFeaturizer):
    """
    Creates rolling window aggregation features.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_names(str or list of str): Names of the columns to create the
            rolling window features on.
        window_size(int or offset): Size of the moving window. This is the
            number of observations used for calculating the statistic. If
            the window size is integer, each window will be a fixed size.
            If it's an offset then this will be the time period of each window.
            Each window will be a variable sized based on the observations
            included in the time-period. This is only valid for datetimelike
            indexes.
        window_args(dict): Additional arguments passed to pandas.rolling.
            See https://pandas.pydata.org/pandas-docs/stable/reference/api/
            pandas.DataFrame.rolling.html
        agg_func(function or str): Function used to aggregate data in the
            rolling window.  If a function, must either work when passed a
            DataFrame or when passed to DataFrame.apply. Default value is
            'mean'. Other commonly used values are 'std', 'quantile'.
        agg_args(dict): Additional arguments passed to the aggregation
            function.
        future_value_available(bool): Whether future values of the input
            columns are available at forecast time. Default value is False.
            It's a read-only property and can not be changed once a featurizer
            is instantiated.
        max_horizon(int): Maximum number of steps ahead to forecast. The step
            unit is the frequency of the data.
            This value is needed to prevent creating features on the
            training data that are not available for the testing data. For
            example, the features and models are created on week 7 to
            forecast week 8 to week 10. It would not make sense to create a
            feature using data from week 8 and week 9, because they are not
            available at the forecast creation  time. Thus, it does not make
            sense to create a feature using data from week 5 and week 6 for
            week 7.
            When future_value_available is False, max_horizon must be
            specified.
        rolling_gap(int): Time interval (in the unit of the data frequency)
            between the rolling window and the current time. If not set and
            if future_value_available is False, rolling_gap is set to be
            equal to max_horizon.
        train_df(pd.DataFrame): Training data needed to compute rolling
            window features on testing data.
            Note: this property must be None when transforming the
            training data, and train_df can only be passed after
            transforming the training data. It's not recommended to save a
            pipeline with train_df not set to None, because it results in a
            large pipeline object, especially when you have multiple
            featurizers requiring the training data at scoring time.
            To set this value on in a pipeline, use the following code
            pipeline.set_params('featurizer_step_name__train_df) = None
            pipeline.set_params('featurizer_step_name__train_df) = train_df
            featurizer_step_name is the name of the featurizer step when
            creating the pipeline.

    Examples:
        This featurizer is scikit-learn compatible and can be used in
        scikit-learn pipelines.
        >>>tsdf = pd.DataFrame({
        ...    'store': [1] * 10 + [2] * 10,
        ...    'date': list(pd.date_range('2011-01-01', '2011-01-10')) +
        ...            list(pd.date_range('2011-01-01', '2011-01-10')),
        ...    'sales': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        ...              11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})

        >>>df_config = {
        ...    'time_col_name': 'date',
        ...    'ts_id_col_names': 'store',
        ...    'target_col_name': 'sales',
        ...    'frequency': 'D',
        ...    'time_format': '%Y-%m-%d'
        ...}

        >>>rolling_window_featurizer = RollingWindowFeaturizer(
        ...    df_config, input_col_names='sales',
        ...    window_size=3, max_horizon=3,
        ...    window_args={'min_periods': 1})
        >>>rolling_window_featurizer.transform(tsdf)
            store       date  sales  sales_mean
        0       1 2011-01-01      1         NaN
        1       1 2011-01-02      2         NaN
        2       1 2011-01-03      3         NaN
        3       1 2011-01-04      4         1.0
        4       1 2011-01-05      5         1.5
        5       1 2011-01-06      6         2.0
        6       1 2011-01-07      7         3.0
        7       1 2011-01-08      8         4.0
        8       1 2011-01-09      9         5.0
        9       1 2011-01-10     10         6.0
        10      2 2011-01-01     11         NaN
        11      2 2011-01-02     12         NaN
        12      2 2011-01-03     13         NaN
        13      2 2011-01-04     14        11.0
        14      2 2011-01-05     15        11.5
        15      2 2011-01-06     16        12.0
        16      2 2011-01-07     17        13.0
        17      2 2011-01-08     18        14.0
        18      2 2011-01-09     19        15.0
        19      2 2011-01-10     20        16.0
    """

    def __init__(
        self,
        df_config,
        input_col_names,
        window_size,
        window_args={},
        agg_func="mean",
        agg_args={},
        future_value_available=False,
        max_horizon=None,
        rolling_gap=None,
        train_df=None,
    ):
        super().__init__(df_config)

        self.input_col_names = input_col_names
        self.window_size = window_size
        self.agg_func = agg_func
        self.agg_args = agg_args
        self._future_value_available = future_value_available
        self.train_df = train_df

        # max_horizon and window_args must be set after future_value_available
        # is set, because they depend on future_value_available
        self.max_horizon = max_horizon
        self.window_args = window_args

        # rolling_gap must be set after max_horizon is set, because it
        # depends on max_horizon
        self.rolling_gap = rolling_gap

        if callable(self.agg_func):
            self._agg_func_name = self.agg_func.__name__
        elif isinstance(self.agg_func, str):
            self._agg_func_name = self.agg_func
        else:
            raise Exception("agg_func must be a function or a string.")

    @property
    def rolling_gap(self):
        return self._rolling_gap

    @rolling_gap.setter
    def rolling_gap(self, val):
        if val is None:
            if self.future_value_available:
                self._rolling_gap = 0
            elif self.max_horizon is not None:
                self._rolling_gap = self.max_horizon
        else:
            self._rolling_gap = val

    def _rolling_window_agg_single_ts(self, input_df, forecast_creation_time):
        """
        Creates rolling window features on a single time series data frame.

        Args:
            input_df(pandas.DataFrame): Input data frame to create rolling
                window features on.
            forecast_creation_time(pandas.datetime): A timestamp specifying
                when the rolling window features are created.
        Returns:
            pandas.DataFrame: Data frame with the time column of input_df
                and rolling window features.
        """
        input_df = convert_to_tsdf(input_df, time_col_name=self.time_col_name, time_format=self.time_format,)

        if not self.future_value_available:
            input_df.loc[input_df.index.get_level_values(0) > forecast_creation_time, self.input_col_names,] = np.nan

        rolling_agg_df = (
            input_df[self.input_col_names]
            .shift(self.rolling_gap)
            .rolling(window=self.window_size, **self.window_args)
            .agg(self.agg_func, **self.agg_args)
        )

        rolling_agg_df.columns = [col + "_" + self._agg_func_name for col in rolling_agg_df.columns]

        return rolling_agg_df


class SameDayOfWeekRollingWindowFeaturizer(BaseRollingWindowFeaturizer):
    """
    Creates rolling window aggregation features based on same day of week data.

    For data with hourly frequency, features are computed by aggregating the
    values of the same hour of day and the same day of earlier weeks.
    For data with daily frequency, features are computed by aggregating the
    values of the same day of earlier weeks.
    For data with weekly frequency, features are computed by aggregating
    values of earlier weeks.

    Args:
        df_config(dict): Configuration of the time series data frame
            to compute features on.
        input_col_names(str or list of str): Names of the columns to create the
            rolling_window feature on.
        window_size(int): Number of weeks used to compute each aggregation.
            The window is on the "older" side of the first week of each window.
            For example, if start_week is 9 and window_size is 4, the output
            feature same_dow_rolling_agg_9 aggregates the same day (and hour)
            values of the 9th, 10th, 11th, and 12th weeks before the current
            week.
        max_horizon(int): Maximum number of steps ahead to forecast. The step
            unit is the frequency of the data.
            This value is needed to prevent creating features on the
            training data that are not available for the testing data. For
            example, the features and models are created on week 7 to
            forecast week 8 to week 10. It would not make sense to create a
            feature using data from week 8 and week 9, because they are not
            available at the forecast creation  time. Thus, it does not make
            sense to create a feature using data from week 5 and week 6 for
            week 7.
            When future_value_available is False, max_horizon must be
            specified.
        future_value_available(bool): Whether future values of the input
            columns are available at forecast time. Default value is False.
            It's a read-only property and can not be changed once a
            featurizer is instantiated.
        start_week(int): Number of weeks between the current week and the first
            week of the first aggregation feature. If not set, it's
            automatically computed based on max_horizon and the data frequency.
        agg_func(function or str): Function used to aggregate data in the
            rolling window.  If a function, must either work when passed a
            DataFrame or when passed to DataFrame.apply. Default value is
            'mean'. Other commonly used values are 'std', 'quantile'.
        agg_count(int): Number of aggregation features to create. Default
            value is 1.
            For example, start_week = 9, window_size=4, and count = 3 will
            create three aggregation of features.
            1) moving_agg_lag_9: aggregate the same day and hour values of
            the 9th, 10th, 11th, and 12th weeks before the current week.
            2) moving_agg_lag_10: aggregate the same day and hour values of the
            10th, 11th, 12th, and 13th weeks before the current week.
            3) moving_agg_lag_11: aggregate the same day and hour values of the
            11th, 12th, 13th, and 14th weeks before the current week.
        agg_args(dict): Additional arguments passed to the aggregation
            function.
        train_df(pd.DataFrame): Training data needed to compute rolling
            window features on testing data.
            Note: this property must be None when transforming the
            training data, and train_df can only be passed after
            transforming the training data. It's not recommended to save a
            pipeline with train_df not set to None, because it results in a
            large pipeline object, especially when you have multiple
            featurizers requiring the training data at scoring time.
            To set this value on in a pipeline, use the following code
            pipeline.set_params('featurizer_step_name__train_df) = None
            pipeline.set_params('featurizer_step_name__train_df) = train_df
            featurizer_step_name is the name of the featurizer step when
            creating the pipeline.
        output_col_suffix(str): Suffix of the output columns. The start week of
            each rolling window is added at the end. Default value
            'same_dow_rolling_agg'.
        round_agg_result(bool): If round the final aggregation result.
            Default value is False.
    Examples:
        This featurizer is scikit-learn compatible and can be used in
        scikit-learn pipelines.
        >>>tsdf = pd.DataFrame({
        ...    'store': [1] * 7 + [2] * 7,
        ...    'date': pd.to_datetime([
        ...        '2017-09-07', '2019-02-14', '2019-02-21',
        ...        '2019-02-28', '2019-03-07', '2019-03-14',
        ...        '2019-03-28'] * 2),
        ...    'sales': [1, 2, 3, 4, 5, 6, 7,
        ...             11, 12, 13, 14, 15, 16, 17]})

        >>>df_config = {
        ...    'time_col_name': 'date',
        ...    'ts_id_col_names': 'store',
        ...    'target_col_name': 'sales',
        ...    'frequency': 'D',
        ...    'time_format': '%Y-%m-%d'
        ...}

        >>>same_dow_rolling_window_featurizer =
        ...    SameDayOfWeekRollingWindowFeaturizer(
        ...    df_config, input_col_names='sales', start_week=2,
        ...    window_size=4, agg_count=2, max_horizon=1,
        ...    output_col_suffix='rolling')
        >>>same_dow_rolling_window_featurizer.transform(tsdf)
                store       date  sales  sales_rolling_2  sales_rolling_3
        0       1 2017-09-07      1              NaN              NaN
        1       1 2019-02-14      2              NaN              NaN
        2       1 2019-02-21      3              NaN              NaN
        3       1 2019-02-28      4              2.0              NaN
        4       1 2019-03-07      5              2.5              2.0
        5       1 2019-03-14      6              3.0              2.5
        6       1 2019-03-28      7              4.5              3.5
        7       2 2017-09-07     11              NaN              NaN
        8       2 2019-02-14     12              NaN              NaN
        9       2 2019-02-21     13              NaN              NaN
        10      2 2019-02-28     14             12.0              NaN
        11      2 2019-03-07     15             12.5             12.0
        12      2 2019-03-14     16             13.0             12.5
        13      2 2019-03-28     17             14.5             13.5
    """

    def __init__(
        self,
        df_config,
        input_col_names,
        window_size,
        max_horizon=None,
        future_value_available=False,
        start_week=1,
        agg_func="mean",
        agg_count=1,
        agg_args={},
        train_df=None,
        output_col_suffix="same_dow_rolling_agg",
        round_agg_result=False,
    ):
        super().__init__(df_config)

        self.input_col_names = input_col_names
        self.window_size = window_size
        self._future_value_available = future_value_available
        self.agg_func = agg_func
        self.agg_count = agg_count
        self.train_df = train_df
        self.agg_args = agg_args
        self.output_col_suffix = output_col_suffix
        self.round_agg_result = round_agg_result

        # max_horizon must be set after future_value_available is set, because
        # it depends on future_value_available
        self.max_horizon = max_horizon
        self.start_week = start_week

    def _rolling_window_agg_single_ts(self, input_df, forecast_creation_time):
        """
        Creates rolling window features on a single time series data frame.

        Args:
            input_df(pandas.DataFrame): Input data frame to create rolling
                window features on.
            forecast_creation_time(pandas.datetime): A timestamp specifying
                when the rolling window features are created.
        Returns:
            pandas.DataFrame: Data frame with the time column of input_df
                and rolling window features.
        """
        input_df = convert_to_tsdf(input_df, time_col_name=self.time_col_name, time_format=self.time_format,)

        output_df = pd.DataFrame({self.time_col_name: input_df.index.get_level_values(0)})
        min_time_stamp = output_df[self.time_col_name].min()
        max_time_stamp = output_df[self.time_col_name].max()

        if not self.future_value_available:
            input_df.loc[input_df.index.get_level_values(0) > forecast_creation_time, self.input_col_names,] = np.nan

        for i in range(self.agg_count):
            week_lag_start = self.start_week + i
            week_lags = [week_lag_start + w for w in range(self.window_size)]

            # Make sure the lag is not too small and not available for the
            # maximum forecasting time point, or too large and not available
            # for any time point
            week_lags = [
                lag
                for lag in week_lags
                if (max_time_stamp - int(lag) * pd.offsets.Week()) >= min_time_stamp
                and (max_time_stamp - int(lag) * pd.offsets.Week()) <= forecast_creation_time
            ]

            tmp_df = pd.DataFrame({"time": input_df.index.get_level_values(0)})
            lag_df = pd.DataFrame(index=input_df.index)
            lag_df.reset_index(inplace=True)
            if len(week_lags) > 0:
                for lag in week_lags:
                    tmp_df["lag_time"] = input_df.index.get_level_values(0) - int(lag) * pd.offsets.Week()
                    lag_df_cur = pd.merge(tmp_df, input_df, how="left", left_on="lag_time", right_index=True,)
                    for col in self.input_col_names:
                        lag_df[col + "_lag_" + str(lag)] = lag_df_cur[col]

                for col in self.input_col_names:
                    lag_cols = [c for c in lag_df.columns if c.startswith(col)]
                    output_col_name = col + "_" + self.output_col_suffix + "_" + str(week_lag_start)

                    output_df[output_col_name] = lag_df[lag_cols].apply(self.agg_func, axis=1, **self.agg_args)
                    if self.round_agg_result:
                        output_df[output_col_name] = round(output_df[output_col_name])
        output_df.set_index(self.time_col_name, inplace=True)
        return output_df
