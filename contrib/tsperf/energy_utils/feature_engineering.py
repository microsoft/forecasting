# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script computes features on the GEFCom2017_D dataset. It is
parameterized so that a selected set of features specified by a feature
configuration list are computed and saved as csv files.
"""
import os
from math import ceil
import pandas as pd
from functools import reduce
from sklearn.pipeline import Pipeline

from fclib.feature_engineering.lag import (
    SameWeekOfYearLagFeaturizer,
    SameDayOfYearLagFeaturizer,
)
from fclib.feature_engineering.temporal import (
    TemporalFeaturizer,
    DayTypeFeaturizer,
    AnnualFourierFeaturizer,
    DailyFourierFeaturizer,
    WeeklyFourierFeaturizer,
)
from fclib.feature_engineering.rolling_window import SameDayOfWeekRollingWindowFeaturizer
from fclib.feature_engineering.normalization import (
    YearNormalizer,
    DateNormalizer,
    DateHourNormalizer,
)

from fclib.dataset.energy.benchmark_paths import DATA_DIR

print("Data directory used: {}".format(DATA_DIR))

pd.set_option("display.max_columns", None)

OUTPUT_DIR = os.path.join(DATA_DIR, "features")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

TRAIN_BASE_FILE = "train_base.csv"
TRAIN_FILE_PREFIX = "train_round_"
TEST_FILE_PREFIX = "test_round_"
NUM_ROUND = 6


# A dictionary mapping each feature name to the featurizer for computing the
# feature
FEATURE_MAP = {
    "temporal": TemporalFeaturizer,
    "annual_fourier": AnnualFourierFeaturizer,
    "weekly_fourier": WeeklyFourierFeaturizer,
    "daily_fourier": DailyFourierFeaturizer,
    "normalized_date": DateNormalizer,
    "normalized_datehour": DateHourNormalizer,
    "normalized_year": YearNormalizer,
    "day_type": DayTypeFeaturizer,
    "recent_load_lag": SameDayOfWeekRollingWindowFeaturizer,
    "recent_temp_lag": SameDayOfWeekRollingWindowFeaturizer,
    "previous_year_load_lag": SameWeekOfYearLagFeaturizer,
    "previous_year_temp_lag": SameDayOfYearLagFeaturizer,
}

# List of features that requires the training data when computing them on the
# testing data
FEATURES_REQUIRE_TRAINING_DATA = [
    "recent_load_lag",
    "recent_temp_lag",
    "previous_year_load_lag",
    "previous_year_temp_lag",
]

# List of features that requires the max_horizon argument to be set
FEATURES_REQUIRE_MAX_HORIZON = [
    "recent_load_lag",
    "recent_temp_lag",
    "previous_year_load_lag",
    "previous_year_temp_lag",
]

# Configuration for computing a scaling factor that captures year over year
# trend. These scaling factors can be used to scale forecasting results if no
# features for capturing the year over year trend are included in the model.
# To compute the load ratios, first, SameDayOfWeekRollingWindowFeaturizer is
# used to compute moving average of the DEMAND of the same hour of day and same
# day of week of seven four-week windows. There is a 10 week gap between the
# latest four-week window and the current week, because of the forecasting
# horizon of this problem.
# Second SameWeekOfYearLagFeaturizer is used to compute the moving average
# features of the same week of year of previous 5 years.
# Finally, the load ratios are computed by dividing the moving average DEMAND
# of previous years by the moving average DEMAND of the current year. The
# idea is that there is a correlation between the DEMAND between the
# current time point and earlier time point, and the ratio between the DEMAND
# of earlier time point of previous years and the current year can be used to
# scale the forecasting results of the current year.
LOAD_RATIO_CONFIG = {
    "same_day_of_week_rolling_args": {
        "window_size": 4,
        "start_week": 10,
        "agg_count": 7,
        "output_col_suffix": "recent_moving_average",
        "round_agg_result": True,
    },
    "same_week_of_year_lag_args": {
        "n_years": 5,
        "week_window": 0,
        "output_col_suffix": "lag",
        "round_agg_result": True,
    },
}


def parse_feature_config(feature_config, feature_map):
    """
    A helper function parsing a feature_config to feature name,
    featurizer class, and arguments to use to initialize the featurizer.
    """
    feature_name = feature_config[0]
    feature_args = feature_config[1]
    featurizer = feature_map[feature_name]

    return feature_name, feature_args, featurizer


def compute_training_features(train_df, df_config, feature_config_list, feature_map, max_horizon):
    """
    Creates a pipeline based on the input feature configuration list and the
    feature_map. Fit the pipeline on the training data and transform
    the training data.

    Args:
        train_df(pd.DataFrame): Training data to fit on and transform.
        df_config(dict): Configuration of the time series data frame to compute
            features on.
        feature_config_list(list of tuples): The first element of each
            feature configuration tuple is the name of the feature,
            which must be a key in feature_map. The second element of each
            feature configuration tuple is a dictionary of arguments to pass
            to the featurizer corresponding the feature name in feature_map.
        feature_map(dict): Maps each feature name (key) to corresponding
            featurizer(value).
        max_horizon(int): Maximum number of steps ahead to forecast.
            The step unit is the frequency of the data.
            This value is needed to prevent creating features on the
            training data that are not available for the testing data. For
            example, the features and models are created on week 7 to
            forecast week 8 to week 10. It would not make sense to create a
            feature using data from week 8 and week 9, because they are not
            available at the forecast creation  time. Thus, it does not make
            sense to create a feature using data from week 5 and week 6 for
            week 7.

    Returns:
        (pd.DataFrame, sklearn.pipeline): (training features, feature
            engineering pipeline fitted on the training data.
    """
    pipeline_steps = []
    for feature_config in feature_config_list:
        feature_name, feature_args, featurizer = parse_feature_config(feature_config, feature_map)
        if feature_name in FEATURES_REQUIRE_MAX_HORIZON:
            feature_args["max_horizon"] = max_horizon
        pipeline_steps.append((feature_name, featurizer(df_config=df_config, **feature_args)))

    feature_engineering_pipeline = Pipeline(pipeline_steps)
    feature_engineering_pipeline_fitted = feature_engineering_pipeline.fit(train_df)
    train_features = feature_engineering_pipeline_fitted.transform(train_df)

    return train_features, feature_engineering_pipeline_fitted


def compute_testing_features(
    test_df, feature_engineering_pipeline, feature_config_list=None, train_df=None,
):

    """
    Computes features on the testing data using a fitted feature engineering
    pipeline.

    Args:
        test_df(pd.DataFrame): Testing data to fit on and transform.
        feature_engineering_pipeline(sklearn.pipeline): A feature engineering
            pipeline fitted on the training data.
        feature_config_list(list of tuples, optional): The first element of
            each feature configuration tuple is the name of the feature,
            which must be a key in feature_map. The second element of each
            feature configuration tuple is a dictionary of arguments to pass
            to the featurizer corresponding the feature name in feature_map.
            A value is required if train_df is not None.
        train_df(pd.DataFrame, optional): Training data needed to compute
            some lag features on testing data.
    Returns:
        pd.DataFrame: Testing features.
    """
    if train_df is not None and feature_config_list is not None:
        train_df_arguments = {}
        for feature_config in feature_config_list:
            feature_step_name = feature_config[0]
            if feature_step_name in FEATURES_REQUIRE_TRAINING_DATA:
                train_df_arguments[feature_step_name + "__train_df"] = train_df
        if len(train_df_arguments) > 0:
            feature_engineering_pipeline.set_params(**train_df_arguments)

    test_features = feature_engineering_pipeline.transform(test_df)

    return test_features


def compute_features_one_round(
    train_base_df,
    train_delta_df,
    test_df,
    df_config,
    feature_config_list,
    feature_map,
    filter_by_month,
    compute_load_ratio=False,
):

    """
    Computes features on one round of training and testing data.
    Args:
        train_base_df(pd.DataFrame): Training data common to all rounds.
        train_delta_df(pd.DataFrame): Additional training data for the
            current round.
        test_df(pd.DataFrame): Testing data of the current round.
        df_config: Configuration of the input dataframes.
        feature_config_list(list of tuples, optional): The first element of
            each feature configuration tuple is the name of the feature,
            which must be a key in feature_map. The second element of each
            feature configuration tuple is a dictionary of arguments to pass
            to the featurizer corresponding the feature name in feature_map.
        feature_map(dict): Maps each feature name (key) to corresponding
            featurizer(value).
        filter_by_month(bool): If filter the training data by the month of
            the testing data.
        compute_load_ratio(bool): If computes a scaling factor that capture
            the year over year trend and can be used to scale the forecasting
            result. If True, load ratios are computed on the testing data
            according to the LOAD_RATIO_CONFIG.
    Returns:
        (pd.DataFrame, pd.DataFrame): (training features, testing features)
    """

    train_round_df = pd.concat([train_base_df, train_delta_df])
    max_train_timestamp = train_round_df[df_config["time_col_name"]].max()
    max_test_timestamp = test_df[df_config["time_col_name"]].max()
    train_test_diff = max_test_timestamp - max_train_timestamp
    max_horizon = ceil(train_test_diff.days * 24 + train_test_diff.seconds / 3600)
    train_features, feature_pipeline = compute_training_features(
        train_round_df, df_config, feature_config_list, feature_map, max_horizon,
    )

    test_features = compute_testing_features(test_df, feature_pipeline, feature_config_list, train_round_df)

    if compute_load_ratio:
        rolling_window_args = LOAD_RATIO_CONFIG["same_day_of_week_rolling_args"]
        previous_years_lag_args = LOAD_RATIO_CONFIG["same_week_of_year_lag_args"]
        same_week_day_hour_rolling_featurizer = SameDayOfWeekRollingWindowFeaturizer(
            df_config, input_col_names=df_config["target_col_name"], max_horizon=max_horizon, **rolling_window_args
        )
        train_df_with_recent_load = same_week_day_hour_rolling_featurizer.transform(train_round_df)
        same_week_day_hour_rolling_featurizer.train_df = train_round_df
        test_df_with_recent_load = same_week_day_hour_rolling_featurizer.transform(test_df)

        time_col_name = df_config["time_col_name"]
        ts_id_col_names = df_config["ts_id_col_names"]
        keep_col_names = [time_col_name]
        if ts_id_col_names is not None:
            if isinstance(ts_id_col_names, list):
                keep_col_names = keep_col_names + ts_id_col_names
            else:
                keep_col_names.append(ts_id_col_names)
        lag_df_list = []
        start_week = rolling_window_args["start_week"]
        end_week = start_week + rolling_window_args["agg_count"]
        for i in range(start_week, end_week):
            col_old = df_config["target_col_name"] + "_" + rolling_window_args["output_col_suffix"] + "_" + str(i)
            col_new = col_old + "_" + previous_years_lag_args["output_col_suffix"]
            col_ratio = "recent_load_ratio_" + str(i)

            same_week_day_hour_lag_featurizer = SameWeekOfYearLagFeaturizer(
                df_config,
                input_col_names=col_old,
                train_df=train_df_with_recent_load,
                max_horizon=max_horizon,
                **previous_years_lag_args
            )

            lag_df = same_week_day_hour_lag_featurizer.transform(test_df_with_recent_load)
            lag_df[col_ratio] = lag_df[col_old] / lag_df[col_new]
            lag_df_list.append(lag_df[keep_col_names + [col_ratio]].copy())

        test_features = reduce(
            lambda left, right: pd.merge(left, right, on=keep_col_names), [test_features] + lag_df_list,
        )

    if filter_by_month:
        test_month = test_features["month_of_year"].values[0]
        train_features = train_features.loc[train_features["month_of_year"] == test_month,].copy()

    train_features.dropna(inplace=True)

    return train_features, test_features


def compute_features(
    train_dir, test_dir, output_dir, df_config, feature_config_list, filter_by_month=True, compute_load_ratio=False,
):
    """
    Computes training and testing features of all rounds on the
    GEFCom2017_D dataset and save as csv files.
    Args:
        train_dir(str): Directory of the training datasets.
        test_dir(str): Directory of the testing datasets.
        output_dir(str): Directory to save the output feature files.
        df_config(dict): Configuration of the dataframes.
        feature_config_list(list of tuples, optional): The first element of
            each feature configuration tuple is the name of the feature,
            which must be a key in feature_map. The second element of each
            feature configuration tuple is a dictionary of arguments to pass
            to the featurizer corresponding the feature name in feature_map.
        filter_by_month(bool): If filter the training data by the month of
            the testing data. Default value is True.
        compute_load_ratio(bool): If computes a scaling factor that capture
            the year over year trend and can be used to scale the forecasting
            result. If True, load ratios are computed on the testing data
            according to the LOAD_RATIO_CONFIG.
    """
    time_col_name = df_config["time_col_name"]

    output_train_dir = os.path.join(output_dir, "train")
    output_test_dir = os.path.join(output_dir, "test")
    if not os.path.isdir(output_train_dir):
        os.mkdir(output_train_dir)
    if not os.path.isdir(output_test_dir):
        os.mkdir(output_test_dir)

    train_base_df = pd.read_csv(os.path.join(train_dir, TRAIN_BASE_FILE), parse_dates=[time_col_name])

    for i in range(1, NUM_ROUND + 1):
        train_file = os.path.join(train_dir, TRAIN_FILE_PREFIX + str(i) + ".csv")
        test_file = os.path.join(test_dir, TEST_FILE_PREFIX + str(i) + ".csv")

        train_delta_df = pd.read_csv(train_file, parse_dates=[time_col_name])
        test_round_df = pd.read_csv(test_file, parse_dates=[time_col_name])

        train_all_features, test_all_features = compute_features_one_round(
            train_base_df,
            train_delta_df,
            test_round_df,
            df_config,
            feature_config_list,
            FEATURE_MAP,
            filter_by_month,
            compute_load_ratio,
        )

        train_output_file = os.path.join(output_dir, "train", TRAIN_FILE_PREFIX + str(i) + ".csv")
        test_output_file = os.path.join(output_dir, "test", TEST_FILE_PREFIX + str(i) + ".csv")

        train_all_features.to_csv(train_output_file, index=False)
        test_all_features.to_csv(test_output_file, index=False)

        print("Round {}".format(i))
        print("Training data size: {}".format(train_all_features.shape))
        print("Testing data size: {}".format(test_all_features.shape))
        print("Minimum training timestamp: {}".format(min(train_all_features[time_col_name])))
        print("Maximum training timestamp: {}".format(max(train_all_features[time_col_name])))
        print("Minimum testing timestamp: {}".format(min(test_all_features[time_col_name])))
        print("Maximum testing timestamp: {}".format(max(test_all_features[time_col_name])))
        print("")
