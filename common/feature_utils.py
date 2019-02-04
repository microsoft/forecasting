"""
This file contains helper functions for creating features for TSPerf
reference implementations and submissions. All functions defined assume 
that there is no missing data.
"""

from datetime import timedelta
import calendar
import pandas as pd
import numpy as np

from utils import is_datetime_like

# 0: Monday, 2: T/W/TR, 4: F, 5:SA, 6: S
WEEK_DAY_TYPE_MAP = {1: 2, 3: 2}    # Map for converting Wednesday and
                                    # Thursday to have the same code as Tuesday
HOLIDAY_CODE = 7
SEMI_HOLIDAY_CODE = 8  # days before and after a holiday

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'


def day_type(datetime_col, holiday_col=None,
             semi_holiday_offset=timedelta(days=1)):
    """
    Convert datetime_col to 7 day types
    0: Monday
    2: Tuesday, Wednesday, and Thursday
    4: Friday
    5: Saturday
    6: Sunday
    7: Holiday
    8: Days before and after a holiday

    Args:
        datetime_col: Datetime column.
        holiday_col: Holiday code column. Default value None.
        semi_holiday_offset: Time difference between the date before (or after)
            the holiday and the holiday. Default value timedelta(days=1).
    
    Returns:
        A numpy array containing converted datatime_col into day types.
    """

    datetype = pd.DataFrame({'DayType': datetime_col.dt.dayofweek})
    datetype.replace({'DayType': WEEK_DAY_TYPE_MAP}, inplace=True)

    if holiday_col is not None:
        holiday_mask = holiday_col > 0
        datetype.loc[holiday_mask, 'DayType'] = HOLIDAY_CODE

        # Create a temporary Date column to calculate dates near the holidays
        datetype['Date'] = pd.to_datetime(datetime_col.dt.date,
                                          format=DATETIME_FORMAT)
        holiday_dates = set(datetype.loc[holiday_mask, 'Date'])

        semi_holiday_dates = \
            [pd.date_range(start=d - semi_holiday_offset,
                           end=d + semi_holiday_offset,
                           freq='D')
             for d in holiday_dates]

        # Flatten the list of lists
        semi_holiday_dates = [d for dates in semi_holiday_dates for d in dates]

        semi_holiday_dates = set(semi_holiday_dates)
        semi_holiday_dates = semi_holiday_dates.difference(holiday_dates)

        datetype.loc[datetype['Date'].isin(semi_holiday_dates), 'DayType'] \
            = SEMI_HOLIDAY_CODE

    return datetype['DayType'].values


def hour_of_day(datetime_col):
    """Returns the hour from a datetime column."""
    return datetime_col.dt.hour


def time_of_year(datetime_col):
    """
    Time of year is a cyclic variable that indicates the annual position and
    repeats each year. It is each year linearly increasing over time going
    from 0 on January 1 at 00:00 to 1 on December 31st at 23:00. The values
    are normalized to be between [0; 1].

    Args:
        datetime_col: Datetime column.
    
    Returns:
        A numpy array containing converted datatime_col into time of year.
    """

    time_of_year = pd.DataFrame({'DayOfYear': datetime_col.dt.dayofyear,
                                 'HourOfDay': datetime_col.dt.hour,
                                 'Year': datetime_col.dt.year})
    time_of_year['TimeOfYear'] = \
        (time_of_year['DayOfYear'] - 1) * 24 + time_of_year['HourOfDay']

    time_of_year['YearLength'] = \
        time_of_year['Year'].apply(
            lambda y: 366 if calendar.isleap(y) else 365)

    time_of_year['TimeOfYear'] = \
        time_of_year['TimeOfYear']/(time_of_year['YearLength'] * 24 - 1)

    return time_of_year['TimeOfYear'].values


def week_of_year(datetime_col):
    """Returns the week from a datetime column."""
    return datetime_col.dt.week


def month_of_year(date_time_col):
    """Returns the month from a datetime column."""
    return date_time_col.dt.month


def day_of_week(date_time_col):
    """Returns the day of week from a datetime column."""
    return date_time_col.dt.dayofweek


def day_of_month(date_time_col):
    """Returns the day of month from a datetime column."""
    return date_time_col.dt.day


def day_of_year(date_time_col):
    """Returns the day of year from a datetime column."""
    return date_time_col.dt.dayofyear


def encoded_month_of_year(month_of_year):
    """
    Create one hot encoding of month of year.
    """
    month_of_year = pd.get_dummies(month_of_year, prefix="MonthOfYear")

    return month_of_year


def encoded_day_of_week(day_of_week):
    """
    Create one hot encoding of day_of_week.
    """
    day_of_week = pd.get_dummies(day_of_week, prefix="DayOfWeek")

    return day_of_week


def encoded_day_of_month(day_of_month):
    """
    Create one hot encoding of day_of_month.
    """
    day_of_month = pd.get_dummies(day_of_month, prefix="DayOfMonth")

    return day_of_month


def encoded_day_of_year(day_of_year):
    """
    Create one hot encoding of day_of_year.
    """
    day_of_year = pd.get_dummies(day_of_year)

    return day_of_year


def encoded_hour_of_day(hour_of_day):
    """
    Create one hot encoding of hour_of_day.
    """
    hour_of_day = pd.get_dummies(hour_of_day, prefix="HourOfDay")

    return hour_of_day


def encoded_week_of_year(week_of_year):
    """
    Create one hot encoding of week_of_year.
    """
    week_of_year = pd.get_dummies(week_of_year, prefix="WeekOfYear")

    return week_of_year


def normalized_current_year(datetime_col, min_year, max_year):
    """
    Temporal feature indicating the position of the year of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    
    Args:
        datetime_col: Datetime column.
        min_year: minimum value of year.
        max_year: maximum value of year.
    
    Returns:
        float: the position of the current year in the min_year:max_year range
    """
    year = datetime_col.dt.year

    if max_year != min_year:
        current_year = (year - min_year)/(max_year - min_year)
    elif max_year == min_year:
        current_year = 0

    return current_year


def normalized_current_date(datetime_col, min_date, max_date):
    """
    Temporal feature indicating the position of the date of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    
    Args:
        datetime_col: Datetime column.
        min_date: minimum value of date.
        max_date: maximum value of date.

    Returns:
        float: the position of the current date in the min_date:max_date range
    """
    date = datetime_col.dt.date
    current_date = (date - min_date).apply(lambda x: x.days)
    
    if max_date != min_date:
        current_date = current_date/(max_date - min_date).days
    elif max_date == min_date:
        current_date = 0

    return current_date


def normalized_current_datehour(datetime_col, min_datehour, max_datehour):
    """
    Temporal feature indicating the position of the hour of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    
    Args:
        datetime_col: Datetime column.
        min_datehour: minimum value of datehour.
        max_datehour: maximum value of datehour.

    Returns:
        float: the position of the current datehour in the min_datehour:max_datehour range
    """
    current_datehour = (datetime_col - min_datehour)\
        .apply(lambda x: x.days*24 + x.seconds/3600)

    max_min_diff = max_datehour - min_datehour
    
    if max_min_diff != 0:
        current_datehour = current_datehour/\
                           (max_min_diff.days * 24 + max_min_diff.seconds/3600)
    elif max_min_diff == 0:
        current_datehour = 0

    return current_datehour


def normalized_series(datetime_col, value_col,
                      output_colname='normalized_series'):
    """
    Creates series normalized to be log of input series devided by global average of each series.
    
    Args:
        datetime_col: Datetime column.
        value_col: Series value column to be normalized.
    
    Returns:
        Normalized value column.
    """
    
    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)
    
    mean_value = df['value'].mean()

    if mean_value != 0:
        df[output_colname] = np.log(df['value']/mean_value)
    elif mean_value == 0:
        df[output_colname] = 0

    return df[[output_colname]]


def normalized_features(datetime_col, value_col,
                        output_colname='normalized_features'):
    """
    Create new features normalized using maximum and minimum.

    Args:
        datetime_col: Datetime column.
        value_col: Feature value column to be normalized.
    
    Returns:
        Normalized value column.
    """

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)
        
    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)
    
    min_value = min(df['value'])
    max_value = max(df['value'])

    if min_value != max_value:
        df[output_colname] = (df['value'] - min_value)/(max_value - min_value)
    elif min_value == max_value:
        df[output_colname] = 0
        
    return df[[output_colname]]


def fourier_approximation(t, n, period):
    """
    Generic helper function to create Fourier Series at different harmonies (n) and periods.

    Args:
        t: Datetime column.
        n: Harmonies, n=0, 1, 2, 3,...
        period: Period of the datetime variable t.
    
    Returns:
        float: Sine component
        float: Cosine component
    """
    x = n * 2 * np.pi * t/period
    x_sin = np.sin(x)
    x_cos = np.cos(x)

    return x_sin, x_cos


def annual_fourier(datetime_col, n_harmonics):
    """
    Creates Annual Fourier Series at different harmonies (n).

    Args:
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...
    
    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """
    day_of_year = datetime_col.dt.dayofyear

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(day_of_year, n, 365.24)

        output_dict['annual_sin_'+str(n)] = sin
        output_dict['annual_cos_'+str(n)] = cos

    return output_dict


def weekly_fourier(datetime_col, n_harmonics):
    """
    Creates Weekly Fourier Series at different harmonies (n).

    Args:
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...
    
    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """
    day_of_week = datetime_col.dt.dayofweek + 1

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(day_of_week, n, 7)

        output_dict['weekly_sin_'+str(n)] = sin
        output_dict['weekly_cos_'+str(n)] = cos

    return output_dict


def daily_fourier(datetime_col, n_harmonics):
    """
    Creates Daily Fourier Series at different harmonies (n).

    Args:
        datetime_col: Datetime column.
        n_harmonics: Harmonies, n=0, 1, 2, 3,...
    
    Returns:
        dict: Output dictionary containing sine and cosine components of
            the Fourier series for all harmonies.
    """
    hour_of_day = datetime_col.dt.hour + 1

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(hour_of_day, n, 24)

        output_dict['daily_sin_'+str(n)] = sin
        output_dict['daily_cos_'+str(n)] = cos

    return output_dict


def same_week_day_hour_lag(datetime_col, value_col, n_years=3,
                           week_window=1, 
                           agg_func='mean',
                           q=None,
                           output_colname='SameWeekHourLag'):
    """
    Creates a lag feature by calculating quantiles, mean and std of values of and
    around the same week, same day of week, and same hour of day, of previous years.

    Args:
        datetime_col: Datetime column.
        value_col: Feature value column to create lag feature from.
        n_years: Number of previous years data to use. Default value 3.
        week_window: Number of weeks before and after the same week to use,
            which should help reduce noise in the data. Default value 1.
        agg_func: Aggregation function to apply on multiple previous values,
            accepted values are 'mean', 'quantile', 'std'. Default value 'mean'.
        q: If agg_func is 'quantile', taking value between 0 and 1.
        output_colname: name of the output lag feature column. 
            Default value 'SameWeekHourLag'.
    
    Returns:
         pandas.DataFrame: data frame containing the newly created lag
            feature as a column.
    """

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)
    min_time_stamp = min(datetime_col)
    max_time_stamp = max(datetime_col)

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    week_lag_base = 52
    week_lag_last_year = list(range(week_lag_base - week_window,
                              week_lag_base + week_window + 1))
    week_lag_all = []
    for y in range(n_years):
        week_lag_all += [x + y * 52 for x in week_lag_last_year]

    week_lag_cols = []
    for w in week_lag_all:
        if (max_time_stamp - timedelta(weeks=w)) >= min_time_stamp:
            col_name = 'week_lag_' + str(w)
            week_lag_cols.append(col_name)

            lag_datetime = df.index.get_level_values(0) - timedelta(weeks=w)
            valid_lag_mask = lag_datetime >= min_time_stamp

            df[col_name] = np.nan

            df.loc[valid_lag_mask, col_name] = \
                df.loc[lag_datetime[valid_lag_mask], 'value'].values

    # Additional aggregation options will be added as needed
    if agg_func == 'mean' and q == None:
        df[output_colname] = round(df[week_lag_cols].mean(axis=1))
    elif agg_func == 'quantile' and q != None:
        df[output_colname] = round(df[week_lag_cols].quantile(q, axis=1))
    elif agg_func == 'std' and q == None:
        df[output_colname] = round(df[week_lag_cols].std(axis=1))

    return df[[output_colname]]


def same_day_hour_lag(datetime_col, value_col, n_years=3,
                      day_window=1, 
                      agg_func='mean',
                      q=None,
                      output_colname='SameDayHourLag'):
    """
    Creates a lag feature by calculating quantiles, mean, and std of values of
    and around the same day of year, and same hour of day, of previous years.
    
    Args:
        datetime_col: Datetime column.
        value_col: Feature value column to create lag feature from.
        n_years: Number of previous years data to use. Default value 3.
        day_window: Number of days before and after the same day to use, 
            which should help reduce noise in the data. Default value 1.
        agg_func: Aggregation function to apply on multiple previous values,
            accepted values are 'mean', 'quantile', 'std'. Default value 'mean'.
        q: If agg_func is 'quantile', taking value between 0 and 1.
        output_colname: name of the output lag feature column. 
            Default value 'SameDayHourLag'.
    
    Returns:
        pandas.DataFrame: data frame containing the newly created lag
            feature as a column.
    """

    if not is_datetime_like(datetime_col):
        datetime_col = pd.to_datetime(datetime_col, format=DATETIME_FORMAT)
    min_time_stamp = min(datetime_col)
    max_time_stamp = max(datetime_col)

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    day_lag_base = 365
    day_lag_last_year = list(range(day_lag_base - day_window,
                                   day_lag_base + day_window + 1))
    day_lag_all = []
    for y in range(n_years):
        day_lag_all += [x + y * 365 for x in day_lag_last_year]

    day_lag_cols = []
    for d in day_lag_all:
        if (max_time_stamp - timedelta(days=d)) >= min_time_stamp:
            col_name = 'day_lag_' + str(d)
            day_lag_cols.append(col_name)

            lag_datetime = df.index.get_level_values(0) - timedelta(days=d)
            valid_lag_mask = lag_datetime >= min_time_stamp

            df[col_name] = np.nan

            df.loc[valid_lag_mask, col_name] = \
                df.loc[lag_datetime[valid_lag_mask], 'value'].values

    # Additional aggregation options will be added as needed
    if agg_func == 'mean' and q == None:
        df[output_colname] = round(df[day_lag_cols].mean(axis=1))
    elif agg_func == 'quantile' and q != None:
        df[output_colname] = round(df[day_lag_cols].quantile(q, axis=1))
    elif agg_func == 'std' and q == None:
        df[output_colname] = round(df[day_lag_cols].std(axis=1))

    return df[[output_colname]]


def same_day_hour_moving_average(datetime_col, value_col, window_size,
                                 start_week, average_count,
                                 forecast_creation_time,
                                 output_col_prefix='moving_average_lag_'):
    """
    Creates moving average features by averaging values of the same day of
    week and same hour of day of previous weeks.

    Args:
        datetime_col: Datetime column
        value_col: Feature value column to create moving average features from.
        window_size: Number of weeks used to compute the average.
        start_week: First week of the first moving average feature.
        average_count: Number of moving average features to create.
        forecast_creation_time: The time point when the feature is created.
            This value is used to prevent using data that are not available
            at forecast creation time to compute features.
        output_col_prefix: Prefix of the output columns. The start week of each 
            moving average feature is added at the end. Default value 'moving_average_lag_'.

    Returns:
        pandas.DataFrame: data frame containing the newly created lag features as
            columns.
    
    For example, start_week = 9, window_size=4, and average_count = 3 will
    create three moving average features.
    1) moving_average_lag_9: average the same day and hour values of the 9th,
    10th, 11th, and 12th weeks before the current week.
    2) moving_average_lag_10: average the same day and hour values of the
    10th, 11th, 12th, and 13th weeks before the current week.
    3) moving_average_lag_11: average the same day and hour values of the
    11th, 12th, 13th, and 14th weeks before the current week.
    """

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    df = df.asfreq('H')

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    df['fct_diff'] = df.index - forecast_creation_time
    df['fct_diff'] = df['fct_diff'].apply(lambda x: x.days*24 + x.seconds/3600)
    max_diff = max(df['fct_diff'])

    for i in range(average_count):
        output_col = output_col_prefix + str(start_week+i)
        week_lag_start = start_week + i
        hour_lags = [(week_lag_start + w) * 24 * 7 for w in range(window_size)]
        hour_lags = [h for h in hour_lags if h > max_diff]
        if len(hour_lags) > 0:
            tmp_df = df[['value']].copy()
            tmp_col_all = []
            for h in hour_lags:
                tmp_col = 'tmp_lag_' + str(h)
                tmp_col_all.append(tmp_col)
                tmp_df[tmp_col] = tmp_df['value'].shift(h)

            df[output_col] = round(tmp_df[tmp_col_all].mean(axis=1))
    df.drop(['fct_diff','value'], inplace=True, axis=1)

    return df


def same_day_hour_moving_quantile(datetime_col, value_col, window_size,
                                 start_week, quantile_count, q,
                                 forecast_creation_time,
                                 output_col_prefix='moving_quatile_lag_'):
    """
    Creates a series of quantiles features by calculating quantiles of values of
    the same day of week and same hour of day of previous weeks.

    Args:
        datetime_col: Datetime column
        value_col: Feature value column to create quantile features from.
        window_size: Number of weeks used to compute the quantile.
        start_week: First week of the first moving quantile feature.
        quantile_count: Number of quantile features to create.
        q: quantile to compute from history values, should be between 0 and 1.
        forecast_creation_time: The time point when the feature is created.
            This value is used to prevent using data that are not available
            at forecast creation time to compute features.
        output_col_prefix: Prefix of the output columns. The start week of each
            moving average feature is added at the end. Default value 'moving_quatile_lag_'.

    Returns:
        pandas.DataFrame: data frame containing the newly created lag features as
            columns.

    For example, start_week = 9, window_size=4, and quantile_count = 3 will
    create three quantiles features.
    1) moving_quantile_lag_9: calculate quantile of the same day and hour values of the 9th,
    10th, 11th, and 12th weeks before the current week.
    2) moving_quantile_lag_10: calculate quantile of average the same day and hour values of the
    10th, 11th, 12th, and 13th weeks before the current week.
    3) moving_quantile_lag_11: calculate quantile of average the same day and hour values of the
    11th, 12th, 13th, and 14th weeks before the current week.
    """

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    df = df.asfreq('H')

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    df['fct_diff'] = df.index - forecast_creation_time
    df['fct_diff'] = df['fct_diff'].apply(lambda x: x.days*24 + x.seconds/3600)
    max_diff = max(df['fct_diff'])

    for i in range(quantile_count):
        output_col = output_col_prefix + str(start_week+i)
        week_lag_start = start_week + i
        hour_lags = [(week_lag_start + w) * 24 * 7 for w in range(window_size)]
        hour_lags = [h for h in hour_lags if h > max_diff]
        if len(hour_lags) > 0:
            tmp_df = df[['value']].copy()
            tmp_col_all = []
            for h in hour_lags:
                tmp_col = 'tmp_lag_' + str(h)
                tmp_col_all.append(tmp_col)
                tmp_df[tmp_col] = tmp_df['value'].shift(h)

        df[output_col] = round(tmp_df[tmp_col_all].quantile(q, axis=1))
        
    df.drop(['fct_diff','value'], inplace=True, axis=1)

    return df


def same_day_hour_moving_std(datetime_col, value_col, window_size,
                             start_week, std_count,
                             forecast_creation_time,
                             output_col_prefix='moving_std_lag_'):
    """
    Creates standard deviation features by calculating std of values of the
    same day of week and same hour of day of previous weeks.

    Args:
        datetime_col: Datetime column
        value_col: Feature value column to create moving std features from.
        window_size: Number of weeks used to compute the std.
        start_week: First week of the first moving std feature.
        std_count: Number of moving std features to create.
        forecast_creation_time: The time point when the feature is created.
            This value is used to prevent using data that are not available at
            forecast creation time to compute features.
        output_col_prefix: Prefix of the output columns. The start week of each
            moving average feature is added at the end. Default value 'moving_std_lag_'.
    
    Returns:
        pandas.DataFrame: data frame containing the newly created lag features as
            columns.

    For example, start_week = 9, window_size=4, and std_count = 3 will
    create three moving std features.
    1) moving_std_lag_9: calculate std of the same day and hour values of the 9th,
    10th, 11th, and 12th weeks before the current week.
    2) moving_std_lag_10: calculate std of the same day and hour values of the
    10th, 11th, 12th, and 13th weeks before the current week.
    3) moving_std_lag_11: calculate std of the same day and hour values of the
    11th, 12th, 13th, and 14th weeks before the current week.
    """

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    df = df.asfreq('H')

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    df['fct_diff'] = df.index - forecast_creation_time
    df['fct_diff'] = df['fct_diff'].apply(lambda x: x.days*24 + x.seconds/3600)
    max_diff = max(df['fct_diff'])

    for i in range(std_count):
        output_col = output_col_prefix + str(start_week+i)
        week_lag_start = start_week + i
        hour_lags = [(week_lag_start + w) * 24 * 7 for w in range(window_size)]
        hour_lags = [h for h in hour_lags if h > max_diff]
        if len(hour_lags) > 0:
            tmp_df = df[['value']].copy()
            tmp_col_all = []
            for h in hour_lags:
                tmp_col = 'tmp_lag_' + str(h)
                tmp_col_all.append(tmp_col)
                tmp_df[tmp_col] = tmp_df['value'].shift(h)

            df[output_col] = round(tmp_df[tmp_col_all].std(axis=1))

    df.drop(['value','fct_diff'], inplace=True, axis=1)

    return df


def same_day_hour_moving_agg(datetime_col, value_col, window_size,
                             start_week, count,
                             forecast_creation_time,
                             agg_func='mean',
                             q=None,
                             output_col_prefix='moving_agg_lag_'):
    """
    Creates a series of aggregation features by calculating mean, quantiles,
    or std of values of the same day of week and same hour of day of previous weeks.

    Args:
        datetime_col: Datetime column
        value_col: Feature value column to create aggregation features from.
        window_size: Number of weeks used to compute the aggregation.
        start_week: First week of the first aggregation feature.
        count: Number of aggregation features to create.
        forecast_creation_time: The time point when the feature is created. 
            This value is used to prevent using data that are not available
            at forecast creation time to compute features.
        agg_func: Aggregation function to apply on multiple previous values, 
            accepted values are 'mean', 'quantile', 'std'.
        q: If agg_func is 'quantile', taking value between 0 and 1.
        output_col_prefix: Prefix of the output columns. The start week of each 
            moving average feature is added at the end. Default value 'moving_agg_lag_'.

    Returns:
        pandas.DataFrame: data frame containing the newly created lag features as
            columns.

    For example, start_week = 9, window_size=4, and count = 3 will
    create three aggregation of features.
    1) moving_agg_lag_9: aggregate the same day and hour values of the 9th,
    10th, 11th, and 12th weeks before the current week.
    2) moving_agg_lag_10: aggregate the same day and hour values of the
    10th, 11th, 12th, and 13th weeks before the current week.
    3) moving_agg_lag_11: aggregate the same day and hour values of the
    11th, 12th, 13th, and 14th weeks before the current week.
    """

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    df = df.asfreq('H')

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    df['fct_diff'] = df.index - forecast_creation_time
    df['fct_diff'] = df['fct_diff'].apply(lambda x: x.days*24 + x.seconds/3600)
    max_diff = max(df['fct_diff'])

    for i in range(count):
        output_col = output_col_prefix + str(start_week+i)
        week_lag_start = start_week + i
        hour_lags = [(week_lag_start + w) * 24 * 7 for w in range(window_size)]
        hour_lags = [h for h in hour_lags if h > max_diff]
        if len(hour_lags) > 0:
            tmp_df = df[['value']].copy()
            tmp_col_all = []
            for h in hour_lags:
                tmp_col = 'tmp_lag_' + str(h)
                tmp_col_all.append(tmp_col)
                tmp_df[tmp_col] = tmp_df['value'].shift(h)

        if agg_func == 'mean' and q == None:
            df[output_col] = round(tmp_df[tmp_col_all].mean(axis=1))        
        elif agg_func == 'quantile' and q != None:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(q, axis=1))
        elif agg_func == 'std' and q == None:
            df[output_col] = round(tmp_df[tmp_col_all].std(axis=1))
            
    df.drop(['fct_diff','value'], inplace=True, axis=1)
    

    return df
