# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd

from fclib.common.plot import plot_predictions_with_history


def test_plot_predictions_with_history(generate_ojdata, generate_data):
    data = pd.read_csv("fclib/tests/resources/ojdatagen.csv")
    pred = generate_data.ojdata(61, 70)
    # implicit assert no-exceptions
    plot_predictions_with_history(
        data,
        pred,
        grain1_unique_vals=[1, 2],
        grain2_unique_vals=[1, 2, 3],
        time_col_name="week",
        target_col_name="logmove",
        grain1_name="store",
        grain2_name="brand",
        min_timestep=min(data.week),
        num_samples=4,
    )
