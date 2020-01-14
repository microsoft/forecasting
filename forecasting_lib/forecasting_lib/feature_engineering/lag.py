from abc import abstractmethod
import pandas as pd
import numpy as np
from collections import Iterable

from tsperf.feature_engineering.base_ts_estimators import BaseTSFeaturizer
from tsperf.feature_engineering.utils import convert_to_tsdf, is_iterable_but_not_string


class BaseLagFeaturizer(BaseTSFeaturizer):
    """
    Base abstract  lag featurizer class for all lag featurizers.
    Child classes need to implement _lag_single_ts, which creates a data
    frame with lag features on a single time series.
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
        if val is None and not self.future_value_available:
            raise Exception(
                "max_horizon must be set when "
                "future_value_available is False"
            )
        self._max_horizon = val

    @abstractmethod
    def _lag_single_ts(self, input_df, forecast_creation_time):
        """
        Creates lag features on a data frame containing a single time series.
        Args:
        input_df(pandas.DataFrame): Input data frame with a time column
            and columns to create lag features on.
        forecast_creation_time(pandas.datetime): A timestamp
            specifying when the lag features are created.

        Returns:
            pandas.DataFrame: Data frame with time column and lag features.
        """
        pass

    def _create_lag_df(self, input_df, lags, offset):
        """
        Creates a data frame with lags.
        Args:
            input_df(pandas.DataFrame): Input data frame with a time index
                and columns to create lag features on.
            lags(int or list of int): Lags to compute.
            offset(pandas.tseries.offsets): Frequency of the shift operation.
                See https://pandas.pydata.org/pandas-docs/stable/user_guide/
                timeseries.html#dateoffset-objects

        Returns:
            pandas.DataFrame: Data frame with time column and lag features.
        """
        if not isinstance(input_df.index, pd.DatetimeIndex):
            raise Exception(
                "The index of input_df must be "
                "pandas.DatetimeIndex. Use utils.convert_to_tsdf "
                "to convert it first."
            )
        input_col_names = input_df.columns
        tmp_df = pd.DataFrame(
            {input_df.index.name: input_df.index.get_level_values(0)}
        )
        lag_df = pd.DataFrame(
            {input_df.index.name: input_df.index.get_level_values(0)}
        )
        for lag in lags:
            tmp_df["lag_time"] = (
                input_df.index.get_level_values(0) - int(lag) * offset
            )
            lag_df_cur = pd.merge(
                tmp_df,
                input_df,
                how="left",
                left_on="lag_time",
                right_index=True,
            )
            for col in input_col_names:
                lag_df[col + "_lag_" + str(lag)] = lag_df_cur[col]

        return lag_df

    def fit(self, X, y=None):
        """
        To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this featurizer.
        """
        return self

    def transform(self, X):
        """
        Creates lag features on a subset of columns in X.

        Args:
             X(pandas.DataFrame): Input data frame to create lag features on.
        Returns:
            pandas.DataFrame: Output data frame with the lag features added
                to the input data frame.
        """
        self._check_config_cols_exist(X)

        col_names = (
            [self.time_col_name] + self.input_col_names + self.ts_id_col_names
        )
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
                # data based on the maximum horizon to forecast on
                max_train_timestamp = time_col.max()
                forecast_creation_time = (
                    max_train_timestamp - self.max_horizon * self._offset
                )
            else:
                forecast_creation_time = time_col.max()
            X_tmp = X[col_names].copy()

        if len(self.ts_id_col_names) == 0:
            X_lag_tmp = self._lag_single_ts(X_tmp, forecast_creation_time)
        else:
            X_lag_tmp = X_tmp.groupby(self.ts_id_col_names).apply(
                lambda g: self._lag_single_ts(g, forecast_creation_time)
            )

            X_lag_tmp.reset_index(inplace=True)

            # ##TODO: Add a boolean argument and refine this
            # X_lag_tmp.fillna(method='ffill', inplace=True)

        if self.train_df is not None:
            X_lag_tmp = X_lag_tmp.loc[
                X_lag_tmp[self.time_col_name] > forecast_creation_time
            ].copy()
        X = pd.merge(X, X_lag_tmp, on=merge_col_names)

        if X.shape[0] == 0:
            raise Exception(
                "The featurizer output is empty. Set the "
                "train_df property of the featurizer to "
                "None if transforming training data."
            )
        return X


class BasePeriodicLagFeaturizer(BaseLagFeaturizer):
    """
    Base abstract class for periodical lag features.
    Periodical lag features are features that from the same time point of
    previous periods, e.g. same week of year of previous years. Multiple
    lags are aggregated to compute a single lag feature. For example,
    when using 5 years of history data, the 5 values from the same week of
    each of the 5 years are aggregate.
    Child classes need to implement _calculate_lags.
    """

    @abstractmethod
    def _calculate_lags(self):
        """
        Computes and returns lags, as integer or list of integers, needed by a
        particular featurizer.
        """
        pass

    def _lag_single_ts(self, input_df, forecast_creation_time):
        """
        Creates lag features on a data frame containing a single time series.

        Args:
        input_df(pandas.DataFrame): Input data frame with a time column
            and columns to create lag features on.
        forecast_creation_time(pandas.datetime): A timestamp
            specifying when the lag features are created.

        Returns:
            pandas.DataFrame: Data frame with time column and lag features.
        """
        input_df = input_df.copy()
        input_df = convert_to_tsdf(
            input_df,
            time_col_name=self.time_col_name,
            time_format=self.time_format,
        )

        output_df = pd.DataFrame(
            {self.time_col_name: input_df.index.get_level_values(0)}
        )
        min_time_stamp = output_df[self.time_col_name].min()
        max_time_stamp = output_df[self.time_col_name].max()

        if not self.future_value_available:
            input_df.loc[
                input_df.index.get_level_values(0) > forecast_creation_time,
                self.input_col_names,
            ] = np.nan

        lag_all = self._calculate_lags()

        lag_all = [
            lag
            for lag in lag_all
            if (max_time_stamp - int(lag) * self._lag_offset) >= min_time_stamp
        ]

        lag_df = self._create_lag_df(
            input_df[self.input_col_names],
            lags=lag_all,
            offset=self._lag_offset,
        )
        for col in self.input_col_names:
            lag_cols = [c for c in lag_df.columns if c.startswith(col)]
            output_col_name = col + "_" + self.output_col_suffix

            output_df[output_col_name] = lag_df[lag_cols].apply(
                self.agg_func, axis=1, **self.agg_args
            )
            if self.round_agg_result:
                output_df[output_col_name] = round(output_df[output_col_name])
        output_df.set_index(self.time_col_name, inplace=True)
        return output_df


class LagFeaturizer(BaseLagFeaturizer):
    """
    Computes regular lag features.
    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_names(str or list of str): Names of the columns to create the
            lag feature on.
        lags(int or list of int): Lags to compute. When
            future_value_available is False, lags must be positive.
        future_value_available(bool): Whether future values of the input
            columns are available at the forecast creation time. Default
            value is False. It's a read-only property and can not be changed
            once a featurizer is instantiated.
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
        train_df(pandas.DataFrame): Training data needed to compute lag
            features on testing data.
            Note: this property must be None when transforming the
            training data, and train_df can only be passed after
            transforming the training data.  It's not recommended to save a
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

        >>>lag_featurizer = LagFeaturizer(df_config, input_col_names='sales',
        ...                               lags=[1, 2, 3],
        ...                               max_horizon=2)
        >>>lag_featurizer.transform(tsdf)
            store       date  sales  sales_lag_1  sales_lag_2  sales_lag_3
        0       1 2011-01-01      1          NaN          NaN          NaN
        1       1 2011-01-02      2          1.0          NaN          NaN
        2       1 2011-01-03      3          2.0          1.0          NaN
        3       1 2011-01-04      4          3.0          2.0          1.0
        4       1 2011-01-05      5          4.0          3.0          2.0
        5       1 2011-01-06      6          5.0          4.0          3.0
        6       1 2011-01-07      7          6.0          5.0          4.0
        7       1 2011-01-08      8          7.0          6.0          5.0
        8       1 2011-01-09      9          8.0          7.0          6.0
        9       1 2011-01-10     10          NaN          8.0          7.0
        10      2 2011-01-01     11          NaN          NaN          NaN
        11      2 2011-01-02     12         11.0          NaN          NaN
        12      2 2011-01-03     13         12.0         11.0          NaN
        13      2 2011-01-04     14         13.0         12.0         11.0
        14      2 2011-01-05     15         14.0         13.0         12.0
        15      2 2011-01-06     16         15.0         14.0         13.0
        16      2 2011-01-07     17         16.0         15.0         14.0
        17      2 2011-01-08     18         17.0         16.0         15.0
        18      2 2011-01-09     19         18.0         17.0         16.0
        19      2 2011-01-10     20          NaN         18.0         17.0
    """

    def __init__(
        self,
        df_config,
        input_col_names,
        lags,
        future_value_available=False,
        max_horizon=None,
        train_df=None,
    ):
        super().__init__(df_config)

        self.input_col_names = input_col_names
        self._future_value_available = future_value_available

        # max_horizon and lags must be set after future_value_available is set,
        # because they depends on it.
        self.max_horizon = max_horizon
        self.lags = lags

        self.train_df = train_df

    @property
    def lags(self):
        return self._lags

    @lags.setter
    def lags(self, val):
        if not isinstance(val, Iterable):
            val = [val]
        if not self.future_value_available:
            for lag in val:
                if lag < 0:
                    raise Exception(
                        "lag can not be negative when "
                        "future_value_available is False"
                    )
        self._lags = val

    def _lag_single_ts(self, input_df, forecast_creation_time):
        """
        Creates lag features on a data frame containing a single time series.
        Args:
        input_df(pandas.DataFrame): Input data frame with a time column
            and columns to create lag features on.
        forecast_creation_time(pandas.datetime): A timestamp
            specifying when the lag features are created.

        Returns:
            pandas.DataFrame: Data frame with time column and lag features.
        """
        input_df = convert_to_tsdf(
            input_df,
            time_col_name=self.time_col_name,
            time_format=self.time_format,
        )

        if not self.future_value_available:
            input_df.loc[
                input_df.index.get_level_values(0) > forecast_creation_time,
                self.input_col_names,
            ] = np.nan

        lag_df = self._create_lag_df(
            input_df[self.input_col_names], lags=self.lags, offset=self._offset
        )

        lag_df.set_index(self.time_col_name, inplace=True)

        return lag_df


class SameWeekOfYearLagFeaturizer(BasePeriodicLagFeaturizer):
    """
    Creates a lag feature based on data from the same week of previous years.

    This feature is useful for data with daily, weekly, and yearly
    seasonalities and has hourly, daily, or weekly frequency. It is computed
    by aggregating values of and around the same week of previous years.
    For hourly data, value of each week is taken from the same hour of day and
    the same day of week.
    For daily data, value of each week is taken from the same day of week.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_names(str or list of str): Names of the columns to create the
            lag feature on.
        train_df(pd.DataFrame): Training data needed to compute lag
            features on testing data.
            Note: this property must be None when transforming the
            training data, and train_df can only be passed after
            transforming the training data.  It's not recommended to save a
            pipeline with train_df not set to None, because it results in a
            large pipeline object, especially when you have multiple
            featurizers requiring the training data at scoring time.
            To set this value on in a pipeline, use the following code
            pipeline.set_params('featurizer_step_name__train_df) = None
            pipeline.set_params('featurizer_step_name__train_df) = train_df
            featurizer_step_name is the name of the featurizer step when
            creating the pipeline.
        n_years(int): Number of previous years data to use. Default value is 3.
        week_window(int): Number of weeks before and after the same week of
            year to use, which should help reduce noise in the data. Default
            value is 1.
        agg_func(function or str): Aggregation function to apply on multiple
            previous values. Default value is 'mean'. Other commonly used
            values are 'std', 'quantile'.
        agg_args(dict): Additional arguments passed to the aggregation
            function.
        future_value_available(bool): Whether future values of the input
            columns are available at the forecast creation time. Default
            value is False. It's a read-only property and can not be changed
            once a featurizer is instantiated.
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
        output_col_suffix(str): Name suffix of the output lag feature column.
            Default value is 'same_woy_lag'.
        round_agg_result(bool): If round the final aggregation result.
            Default value is False.

    Examples:
        This featurizer is scikit-learn compatible and can be used in
        scikit-learn pipelines.
        Note that in the example below, the lag feature of '2019-03-28' is
        the average of the first 6 values, which correspond to two 3-week
        windows around the same time of previous two years.
        >>>tsdf = pd.DataFrame({
        ...    'store': [1] * 7 + [2] * 7,
        ...    'date': pd.to_datetime([
        ...        '2017-03-23', '2017-03-30', '2017-04-06',
        ...        '2018-03-22', '2018-03-29', '2018-04-05',
        ...        '2019-03-28'] * 2),
        ...    'sales': [1, 2, 3, 4, 5, 6, 7,
        ...              11, 12, 13, 14, 15, 16, 17]})

        >>>df_config = {
        ...    'time_col_name': 'date',
        ...    'ts_id_col_names': 'store',
        ...    'target_col_name': 'sales',
        ...    'frequency': 'D',
        ...    'time_format': '%Y-%m-%d'
        ...}

        >>>same_woy_lag_featurizer = SameWeekOfYearLagFeaturizer(
        ...    df_config,  input_col_names='sales', n_years=2, week_window=1,
        ...    max_horizon=1)
        >>>same_woy_lag_featurizer.transform(tsdf)
            store       date  sales  sales_same_woy_lag
        0       1 2017-03-23      1                 NaN
        1       1 2017-03-30      2                 NaN
        2       1 2017-04-06      3                 NaN
        3       1 2018-03-22      4                 1.5
        4       1 2018-03-29      5                 2.0
        5       1 2018-04-05      6                 2.5
        6       1 2019-03-28      7                 3.5
        7       2 2017-03-23     11                 NaN
        8       2 2017-03-30     12                 NaN
        9       2 2017-04-06     13                 NaN
        10      2 2018-03-22     14                11.5
        11      2 2018-03-29     15                12.0
        12      2 2018-04-05     16                12.5
        13      2 2019-03-28     17                13.5
    """

    def __init__(
        self,
        df_config,
        input_col_names,
        train_df=None,
        n_years=3,
        week_window=1,
        agg_func="mean",
        agg_args={},
        future_value_available=False,
        max_horizon=None,
        output_col_suffix="same_woy_lag",
        round_agg_result=False,
    ):
        super().__init__(df_config)

        self.input_col_names = input_col_names
        self.n_years = n_years
        self.week_window = week_window
        self.agg_func = agg_func
        self.agg_args = agg_args
        self._future_value_available = future_value_available
        self.output_col_suffix = output_col_suffix
        self.train_df = train_df
        self.round_agg_result = round_agg_result

        # max_horizon must be set after future_value_available is set,
        # because it depends on it.
        self.max_horizon = max_horizon

        self._lag_offset = pd.offsets.Week()

    def _calculate_lags(self):
        """Calculate of list of numbers of weeks to lag."""
        week_lag_base = 52
        week_lag_last_year = list(
            range(
                week_lag_base - self.week_window,
                week_lag_base + self.week_window + 1,
            )
        )
        week_lag_all = []
        for i in range(self.n_years):
            week_lag_all += [j + i * 52 for j in week_lag_last_year]

        return week_lag_all


class SameDayOfYearLagFeaturizer(BasePeriodicLagFeaturizer):

    """
    Creates a lag feature based on data from the same day of previous years.

    This feature is useful for data with daily and yearly seasonalities and
    has hourly or daily frequency. It is computed by aggregating values of and
    around the same day of year of previous years. For hourly data, value of
    each day is taken from the same hour of day.

    Args:
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        input_col_names(str or list of str): Names of the columns to create the
            lag feature on.
        train_df(pd.DataFrame): Training data needed to compute lag
            features on testing data.
            Note: this property must be None when transforming the
            training data, and train_df can only be passed after
            transforming the training data.  It's not recommended to save a
            pipeline with train_df not set to None, because it results in a
            large pipeline object, especially when you have multiple
            featurizers requiring the training data at scoring time.
            To set this value on in a pipeline, use the following code
            pipeline.set_params('featurizer_step_name__train_df) = None
            pipeline.set_params('featurizer_step_name__train_df) = train_df
            featurizer_step_name is the name of the featurizer step when
            creating the pipeline.
        n_years(int): Number of previous years data to use. Default value is 3.
        day_window(int): Number of days before and after the same day  of
            year to use, which should help reduce noise in the data. Default
            value is 1.
        agg_func(function or str): Aggregation function to apply on multiple
            previous values. Default value is 'mean'. Other commonly used
            values are 'std', 'quantile'.
        agg_args(dict): Additional arguments passed to the aggregation
            function.
        future_value_available(bool): Whether future values of the input
            columns are available at the forecast creation time. Default
            value is False. It's a read-only property and can not be changed
            once a featurizer is instantiated.
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
        output_col_suffix(str): Name suffix of the output lag feature column.
            Default value is 'same_doy_lag'.
        round_agg_result(bool): If round the final aggregation result.
            Default value is False.
    Examples:
        This featurizer is scikit-learn compatible and can be used in
        scikit-learn pipelines.
        Note that in the example below, the lag feature of '2019-03-28' is
        the average of the first 6 values, which correspond to two 3-day
        windows around the same time of previous two years.
        >>>tsdf = pd.DataFrame({
        ...    'store': [1] * 7 + [2] * 7,
        ...    'date': pd.to_datetime([
        ...        '2017-03-27', '2017-03-28', '2017-03-29',
        ...        '2018-03-27', '2018-03-28', '2018-03-29',
        ...        '2019-03-28'] * 2),
        ...    'sales': [1, 2, 3, 4, 5, 6, 7,
        ...              11, 12, 13, 14, 15, 16, 17]})

        >>>df_config = {
        ...    'time_col_name': 'date',
        ...    'ts_id_col_names': 'store',
        ...    'target_col_name': 'sales',
        ...    'frequency': 'D',
        ...    'time_format': '%Y-%m-%d'
        ...}

        >>>same_doy_lag_featurizer = SameDayOfYearLagFeaturizer(
        ...    df_config,  input_col_names='sales', n_years=2, day_window=1,
        ...    max_horizon=1)

        >>>same_doy_lag_featurizer.transform(tsdf)
                store       date  sales  sales_same_doy_lag
        0       1 2017-03-27      1                 NaN
        1       1 2017-03-28      2                 NaN
        2       1 2017-03-29      3                 NaN
        3       1 2018-03-27      4                 1.5
        4       1 2018-03-28      5                 2.0
        5       1 2018-03-29      6                 2.5
        6       1 2019-03-28      7                 3.5
        7       2 2017-03-27     11                 NaN
        8       2 2017-03-28     12                 NaN
        9       2 2017-03-29     13                 NaN
        10      2 2018-03-27     14                11.5
        11      2 2018-03-28     15                12.0
        12      2 2018-03-29     16                12.5
        13      2 2019-03-28     17                13.5
    """

    def __init__(
        self,
        df_config,
        input_col_names,
        train_df=None,
        n_years=3,
        day_window=1,
        agg_func="mean",
        agg_args={},
        future_value_available=False,
        max_horizon=None,
        output_col_suffix="same_doy_lag",
        round_agg_result=False,
    ):
        super().__init__(df_config)
        self.input_col_names = input_col_names
        self.n_years = n_years
        self.day_window = day_window
        self.agg_func = agg_func
        self.agg_args = agg_args
        self._future_value_available = future_value_available
        self.output_col_suffix = output_col_suffix
        self.train_df = train_df
        self.round_agg_result = round_agg_result

        # max_horizon must be set after future_value_available is set,
        # because it depends on it.
        self.max_horizon = max_horizon

        self._lag_offset = pd.offsets.Day()

    def _calculate_lags(self):
        """Calculates a list of numbers of days to lag."""
        day_lag_base = 365
        day_lag_last_year = list(
            range(
                day_lag_base - self.day_window,
                day_lag_base + self.day_window + 1,
            )
        )
        day_lag_all = []
        for i in range(self.n_years):
            day_lag_all += [j + i * 365 for j in day_lag_last_year]

        return day_lag_all
