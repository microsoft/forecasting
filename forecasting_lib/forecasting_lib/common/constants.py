# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os


def repo_path():
    """Return the path of the forecasting repo"""

    num_levels = 4
    path = os.path.abspath(__file__)
    for i in range(num_levels):
        path = os.path.dirname(path)
    return path
