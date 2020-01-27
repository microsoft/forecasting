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
    title="Prediction results for a few sample time series",
    x_label="time step",
    y_label="target value",
    random_seed=2,
):
    """Plot prediction results with historical values

    Args:
        predictions (Dataframe): Prediction results (results)
        history (Dataframe): Historical values (sales)
        grain1_unique_vals (List): store_list
        grain2_unique_vals (List): brand_list
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
                col.set_title("{} {} {} {}".format(grain1_name, grain1_id, grain2_name, grain2_id))
                col.xaxis.set_major_locator(MaxNLocator(integer=True))
                col.set_xlabel(x_label)
                col.set_ylabel(y_label)
                col.legend(labels=["actual", "predicted"])
                sample_id += 1
            else:
                col.axis("off")
    plt.tight_layout()
