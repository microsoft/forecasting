"""
This file contains helper functions for creating features for TSPerf
reference implementations and submissions.
"""

from datetime import timedelta
import calendar
import pandas as pd
import numpy as np
import math

from .utils import is_datetime_like

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
    return datetime_col.dt.hour


def time_of_year(datetime_col):
    """
    Time of year is a cyclic variable that indicates the annual position and
    repeats each year. It is each year linearly increasing over time going
    from 0 on January 1 at 00:00 to 1 on December 31st at 23:00. The values
    are normalized to be between [0; 1].
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
    return datetime_col.dt.week


def month_of_year(date_time_col):
    return date_time_col.dt.month


def day_of_week(date_time_col):
    return date_time_col.dt.dayofweek


def day_of_month(date_time_col):
    return date_time_col.dt.day


def day_of_year(date_time_col):
    return date_time_col.dt.dayofyear


def encoded_month_of_year(month_of_year):
    """
    Create one hot encoding of month of year.
    """
    encoded_month_of_year = pd.get_dummies(month_of_year)

    return encoded_month_of_year


def encoded_day_of_week(day_of_week):
    """
    Create one hot encoding of day_of_week.
    """
    encoded_day_of_week = pd.get_dummies(day_of_week)

    return encoded_day_of_week


def encoded_day_of_month(day_of_month):
    """
    Create one hot encoding of day_of_month.
    """
    encoded_day_of_month = pd.get_dummies(day_of_month)

    return encoded_day_of_month


def encoded_day_of_year(day_of_year):
    """
    Create one hot encoding of day_of_year.
    """
    encoded_day_of_month = pd.get_dummies(day_of_year)

    return encoded_day_of_year


def encoded_hour_week_diff(hour_of_day, week_of_year):
    """
    Create one hot encoding of hour_of_day - week_of_year.
    """
    encoded_hour_week_diff = pd.get_dummies(hour_of_day - week_of_year)

    return encoded_hour_week_diff

def normalized_current_year(datetime_col, min_year, max_year):
    """
    Temporal feature indicating the position of the year of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    """
    year = datetime_col.dt.year
    current_year = (year - min_year)/(max_year - min_year)

    return current_year


def normalized_current_date(datetime_col, min_date, max_date):
    """
    Temporal feature indicating the position of the date of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    """
    date = datetime_col.dt.date
    current_date = (date - min_date).apply(lambda x: x.days)

    current_date = current_date/(max_date - min_date).days

    return current_date


def normalized_current_datehour(datetime_col, min_datehour, max_datehour):
    """
    Temporal feature indicating the position of the hour of a record in the
    entire time period under consideration, normalized to be between 0 and 1.
    """
    current_datehour = (datetime_col - min_datehour)\
        .apply(lambda x: x.days*24 + x.seconds/3600)

    max_min_diff = max_datehour - min_datehour

    current_datehour = current_datehour/\
                       (max_min_diff.days * 24 + max_min_diff.seconds/3600)

    return current_datehour


def normalized_series(datetime_col, value_col, output_colname='normalized_series'):
    """
    Create series normalized to be log of previous series devided by global average of each series.
    
    :param datetime_col: Datetime column
    :param value_col:
        Series value column to be normalized 
    """
    
    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    df[['value']] = math.log10(df[['value']]/df[['value']].mean(axis=1))

    return df


def normalized_features(datetime_col, value_col, output_colname='normalized_features'):
    """
    Create Load and DryBulb temperature normalized using maximum and minimum 

    :param datetime_col: Datetime column
    :param value_col:
        Feature value column to be normalized
    """

    df = pd.DataFrame({'Datetime': datetime_col, 'value': value_col})
    df.set_index('Datetime', inplace=True)

    if not df.index.is_monotonic:
        df.sort_index(inplace=True)

    df[['value']] = (df[['value']] - min(df[['value']])/(max(df[['value']]) - min(df[['value']]))

    return df


def fourier_approximation(t, n, period):
    """
    Generic helper function for create Fourier Series at different
    harmonies(n) and periods.
    """
    x = n * 2 * np.pi * t/period
    x_sin = np.sin(x)
    x_cos = np.cos(x)

    return x_sin, x_cos


def annual_fourier(datetime_col, n_harmonics):
    day_of_year = datetime_col.dt.dayofyear

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(day_of_year, n, 365.24)

        output_dict['annual_sin_'+str(n)] = sin
        output_dict['annual_cos_'+str(n)] = cos

    return output_dict


def weekly_fourier(datetime_col, n_harmonics):
    day_of_week = datetime_col.dt.dayofweek + 1

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(day_of_week, n, 7)

        output_dict['weekly_sin_'+str(n)] = sin
        output_dict['weekly_cos_'+str(n)] = cos

    return output_dict


def daily_fourier(datetime_col, n_harmonics):
    hour_of_day = datetime_col.dt.hour + 1

    output_dict = {}
    for n in range(1, n_harmonics+1):
        sin, cos = fourier_approximation(hour_of_day, n, 24)

        output_dict['daily_sin_'+str(n)] = sin
        output_dict['daily_cos_'+str(n)] = cos

    return output_dict


def same_week_day_hour_lag(datetime_col, value_col, n_years=3,
                           week_window=1, 
                           agg_func=['mean', 'quantile', 'std'],
                           q=[None, .01, .05, .2, .5, .8, .95, .99],
                           output_colname='SameWeekHourLag'):
    """
    Create a lag feature by calculating quantiles, mean and std of values of and around the same week,
    same day of week, and same hour of day, of previous years.
    :param datetime_col: Datetime column
    :param value_col: Feature value column to create lag feature from
    :param n_years: Number of previous years data to use
    :param week_window:
        Number of weeks before and after the same week to
        use, which should help reduce noise in the data
    :param agg_func: aggregation function to apply on multiple previous values
    :param q: quantile value 
    :param output_colname: name of the output lag feature column
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
    elif agg_func == 'quantile' and q == .01:
        df[output_colname] = round(df[week_lag_cols].quantile(.01, axis=1))
    elif agg_func == 'quantile' and q == .05:
        df[output_colname] = round(df[week_lag_cols].quantile(.05, axis=1))
    elif agg_func == 'quantile' and q == .2:
        df[output_colname] = round(df[week_lag_cols].quantile(.2, axis=1))
    elif agg_func == 'quantile' and q == .5:
        df[output_colname] = round(df[week_lag_cols].quantile(.5, axis=1))
    elif agg_func == 'quantile' and q == .8:
        df[output_colname] = round(df[week_lag_cols].quantile(.8, axis=1))
    elif agg_func == 'quantile' and q == .95:
        df[output_colname] = round(df[week_lag_cols].quantile(.95, axis=1))
    elif agg_func == 'quantile' and q == .99:
        df[output_colname] = round(df[week_lag_cols].quantile(.99, axis=1))
    elif agg_func == 'std' and q == None:
        df[output_colname] = round(df[week_lag_cols].std(axis=1))

    return df[[output_colname]]


def same_day_hour_lag(datetime_col, value_col, n_years=3,
                      day_window=1, 
                      agg_func=['mean', 'quantile', 'std'],
                      q=[None, .01, .05, .2, .5, .8, .95, .99],
                      output_colname='SameDayHourLag'):
    """
    Create a lag feature by calculating quantiles, mean, and std of values of and around the same day of
    year, and same hour of day, of previous years.
    :param datetime_col: Datetime column
    :param value_col: Feature value column to create lag feature from
    :param n_years: Number of previous years data to use
    :param day_window:
        Number of days before and after the same day to
        use, which should help reduce noise in the data
    :param agg_func: aggregation function to apply on multiple previous values
    :param q: quantile value
    :param output_colname: name of the output lag feature column
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
        df[output_colname] = round(df[week_lag_cols].mean(axis=1))
    elif agg_func == 'quantile' and q == .01:
        df[output_colname] = round(df[week_lag_cols].quantile(.01, axis=1))
    elif agg_func == 'quantile' and q == .05:
        df[output_colname] = round(df[week_lag_cols].quantile(.05, axis=1))
    elif agg_func == 'quantile' and q == .2:
        df[output_colname] = round(df[week_lag_cols].quantile(.2, axis=1))
    elif agg_func == 'quantile' and q == .5:
        df[output_colname] = round(df[week_lag_cols].quantile(.5, axis=1))
    elif agg_func == 'quantile' and q == .8:
        df[output_colname] = round(df[week_lag_cols].quantile(.8, axis=1))
    elif agg_func == 'quantile' and q == .95:
        df[output_colname] = round(df[week_lag_cols].quantile(.95, axis=1))
    elif agg_func == 'quantile' and q == .99:
        df[output_colname] = round(df[week_lag_cols].quantile(.99, axis=1))
    elif agg_func == 'std' and q == None:
        df[output_colname] = round(df[week_lag_cols].std(axis=1))

    return df[[output_colname]]


def same_day_hour_moving_average(datetime_col, value_col, window_size,
                                 start_week, average_count,
                                 forecast_creation_time,
                                 output_col_prefix='moving_average_lag_'):
    """
    Create a moving average features by averaging values of the same day of
    week and same hour of day of previous weeks.

    :param datetime_col: Datetime column
    :param value_col:
        Feature value column to create moving average features
        from.
    :param window_size: Number of weeks used to compute the average.
    :param start_week: First week of the first moving average feature.
    :param average_count: Number of moving average features to create.
    :param forecast_creation_time:
        The time point when the feature is created. This value is used to
        prevent using data that are not available at forecast creation time
        to compute features.
    :param output_col_prefix:
        Prefix of the output columns. The start week of each moving average
        feature is added at the end.

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
    df.drop('value', inplace=True, axis=1)

    return df


def same_day_hour_moving_quatile(datetime_col, value_col, window_size,
                                 start_week, quatile_count, 
                                 q=[.01, .05, .2, .5, .8, .95, .99],
                                 forecast_creation_time,
                                 output_col_prefix='moving_quatile_lag_'):
    """
    Create a series of quatiles of features by calculating quatiles of values of the same day of
    week and same hour of day of previous weeks.

    :param datetime_col: Datetime column
    :param value_col:
        Feature value column to create moving average features
        from.
    :param window_size: Number of weeks used to compute the average.
    :param start_week: First week of the first moving average feature.
    :param quantile_count: Number of quantiles of features to create.
    :param q: quantile values.
    :param forecast_creation_time:
        The time point when the feature is created. This value is used to
        prevent using data that are not available at forecast creation time
        to compute features.
    :param output_col_prefix:
        Prefix of the output columns. The start week of each moving average
        feature is added at the end.

    For example, start_week = 9, window_size=4, and quantile_count = 3 will
    create three quantiles of features.
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

        if q == .01:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.01, axis=1))
        elif q == .05:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.05, axis=1))
        elif q == .2:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.2, axis=1))
        elif q == .5:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.5, axis=1))
        elif q == .8:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.8, axis=1))
        elif q == .95:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.95, axis=1))
        elif q == .99:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.99, axis=1))

    df.drop('value', inplace=True, axis=1)

    return df


def same_day_hour_moving_std(datetime_col, value_col, window_size,
                             start_week, std_count,
                             forecast_creation_time,
                             output_col_prefix='moving_std_lag_'):
    """
    Create a standard deviation of features by calculating std of values of the same day of
    week and same hour of day of previous weeks.

    :param datetime_col: Datetime column
    :param value_col:
        Feature value column to create moving average features
        from.
    :param window_size: Number of weeks used to compute the average.
    :param start_week: First week of the first moving average feature.
    :param std_count: Number of moving average features to create.
    :param forecast_creation_time:
        The time point when the feature is created. This value is used to
        prevent using data that are not available at forecast creation time
        to compute features.
    :param output_col_prefix:
        Prefix of the output columns. The start week of each moving average
        feature is added at the end.

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
    df.drop('value', inplace=True, axis=1)

    return df


def same_day_hour_moving_agg(datetime_col, value_col, window_size,
                             start_week, count,
                             agg_func=['mean', 'quantile', 'std'],
                             q=[None, .01, .05, .2, .5, .8, .95, .99],
                             forecast_creation_time,
                             output_col_prefix='moving_agg_lag_'):
    """
    Create a series of aggregation of features by calculating mean, quantiles, 
    and std of values of the same day of week and same hour of day of previous weeks.

    :param datetime_col: Datetime column
    :param value_col:
        Feature value column to create a series of aggregation of features
        from.
    :param window_size: Number of weeks used to compute the aggregation.
    :param start_week: First week of the first aggregation of feature.
    :param count: Number of aggregation of features to create.
    :param forecast_creation_time:
        The time point when the feature is created. This value is used to
        prevent using data that are not available at forecast creation time
        to compute features.
    :param output_col_prefix:
        Prefix of the output columns. The start week of each moving average
        feature is added at the end.

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
        elif agg_func == 'quantile' and q == .01:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.01, axis=1))
        elif agg_func == 'quantile' and q == .05:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.05, axis=1))
        elif agg_func == 'quantile' and q == .2:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.2, axis=1))
        elif agg_func == 'quantile' and q == .5:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.5, axis=1))
        elif agg_func == 'quantile' and q == .8:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.8, axis=1))
        elif agg_func == 'quantile' and q == .95:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.95, axis=1))
        elif agg_func == 'quantile' and q == .99:
            df[output_col] = round(tmp_df[tmp_col_all].quantile(.99, axis=1))
        elif agg_func == 'std' and q == None:
            df[output_col] = round(tmp_df[tmp_col_all].std(axis=1))
            
    df.drop('value', inplace=True, axis=1)

    return df
