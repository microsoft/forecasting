# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import numpy as np


def MAPE(predictions, actuals):
    """
    Implements Mean Absolute Percent Error (MAPE).

    Args:
        predictions (pandas.Series or list or numpy.array): a vector of predicted values.
        actuals (pandas.Series or list or numpy.array): a vector of actual values.

    Returns:
        MAPE value
    """
    predictions, actuals = np.array(predictions), np.array(actuals)
    return np.mean(np.abs((actuals - predictions) / actuals))


def sMAPE(predictions, actuals):
    """
    Implements Symmetric Mean Absolute Percent Error (sMAPE).

    Args:
        predictions (pandas.Series or list or numpy.array): a vector of predicted values.
        actuals (pandas.Series or list or numpy.array): a vector of actual values.

    Returns:
        sMAPE value
    """
    predictions, actuals = np.array(predictions), np.array(actuals)
    return np.mean(np.abs(predictions - actuals) / (np.abs(predictions) + np.abs(actuals)))


def pinball_loss(predictions, actuals, q):
    """
    Implements pinball loss evaluation function.

    Args:
        predictions (pandas.Series): a vector of predicted values.
        actuals (pandas.Series): a vector of actual values.
        q (float): The quantile to compute the loss on, the value should
            be between 0 and 1.

    Returns:
        A pandas Series of pinball loss values for each prediction.
    """
    zeros = pd.Series([0] * len(predictions))
    return (predictions - actuals).combine(zeros, max) * (1 - q) + (actuals - predictions).combine(zeros, max) * q
