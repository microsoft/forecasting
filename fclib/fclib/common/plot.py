# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import math
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_predictions_with_history(
    predictions,
    history,
    grain1_unique_vals,
    grain2_unique_vals,
    time_col_name,
    target_col_name,
    grain1_name="grain1",
    grain2_name="grain2",
    min_timestep=1,
    num_samples=4,
    predict_at_timestep=1,
    line_at_predict_time=False,
    title="Prediction results for a few sample time series",
    x_label="time step",
    y_label="target value",
    random_seed=2,
):
    """Plot prediction results with historical values

    Args:
        predictions (pd.DataFrame): Prediction results with a time step column (e.g., week_index), a 
            forecasted value column (e.g., forecasted sales of each store-brand), and two columns that 
            identify each individual time series (e.g., store_id and brand_id) 
        history (pd.Dataframe): A dataframe containing historical values of the prediction target, a 
            time step column, and two columns that specify each time series
        grain1_unique_vals (list): Unique values of the 1st column indicating the granularity of
            the time series data (e.g, store_list)
        grain2_unique_vals (list): Unique values of the 2nd column indicating the granularity of
            the time series data (e.g., brand_list)
        time_col_name (str): Name of the time step column (e.g., week_index)
        target_col_name (str): Name of the forecast target column (e.g., unit_sales)
        grain1_name (str): Name of the 1st column indicating the time series graunularity
        grain2_name (str): Name of the 2nd column indicating the time series graunularity
        min_timestep (int): Minimum time steps in the plots
        num_samples (int): Number of samples from all the time series (each combination of 
            grain1 column and grain2 column gives an individual time series)
        predict_at_timestep (int): Time step at which the forecasts are made
        line_at_predict_time (bool): Whether to add a vertical line indicating the time step 
            when the forecasts are made
        title (str): Overall title of the plots 
        x_label (str): Label of the x-axis of the plots
        y_label (str): Label of the y-axis of the plots
        random_seed (int): Random seed used for sampling the time series
    """

    random.seed(random_seed)

    grain_combinations = list(itertools.product(grain1_unique_vals, grain2_unique_vals))
    sample_grain_combinations = random.sample(grain_combinations, num_samples)
    max_timestep = max(predictions[time_col_name].unique())

    fig, axes = plt.subplots(nrows=math.ceil(num_samples / 2), ncols=2, figsize=(15, 5 * math.ceil(num_samples / 2)))
    if axes.ndim == 1:
        axes = np.reshape(axes, (1, axes.shape[0]))
    fig.suptitle(title, y=1.02, fontsize=20)

    sample_id = 0
    for row in axes:
        for col in row:
            if sample_id < len(sample_grain_combinations):
                [grain1_id, grain2_id] = sample_grain_combinations[sample_id]
                history_sub = history.loc[
                    (history[grain1_name] == grain1_id)
                    & (history[grain2_name] == grain2_id)
                    & (history[time_col_name] <= max_timestep)
                    & (history[time_col_name] >= min_timestep)
                ]
                predictions_sub = predictions.loc[
                    (predictions[grain1_name] == grain1_id)
                    & (predictions[grain2_name] == grain2_id)
                    & (predictions[time_col_name] >= min_timestep)
                ]
                col.plot(history_sub[time_col_name], history_sub[target_col_name], marker="o")
                col.plot(
                    predictions_sub[time_col_name],
                    predictions_sub[target_col_name],
                    linestyle="--",
                    marker="^",
                    color="red",
                )
                if line_at_predict_time:
                    col.axvline(x=predict_at_timestep, linestyle="--")
                col.set_title("{} {} {} {}".format(grain1_name, grain1_id, grain2_name, grain2_id))
                col.xaxis.set_major_locator(MaxNLocator(integer=True))
                col.set_xlabel(x_label)
                col.set_ylabel(y_label)
                col.legend(labels=["actual", "predicted"])
                sample_id += 1
            else:
                col.axis("off")
    plt.tight_layout()
