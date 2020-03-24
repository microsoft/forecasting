# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd


def MAPE(predictions, actuals):
    """
    Implements Mean Absolute Percent Error (MAPE).

    Args:
        predictions (array like): a vector of predicted values.
        actuals (array like): a vector of actual values.

    Returns:
        numpy.float: MAPE value
    """
    if not (isinstance(actuals, pd.Series) and isinstance(predictions, pd.Series)):
        predictions, actuals = pd.Series(predictions), pd.Series(actuals)

    return ((predictions - actuals).abs() / actuals).mean()


def sMAPE(predictions, actuals):
    """
    Implements Symmetric Mean Absolute Percent Error (sMAPE).

    Args:
        predictions (array like): a vector of predicted values.
        actuals (array like): a vector of actual values.

    Returns:
        numpy.float: sMAPE value
    """
    if not (isinstance(actuals, pd.Series) and isinstance(predictions, pd.Series)):
        predictions, actuals = pd.Series(predictions), pd.Series(actuals)

    return ((predictions - actuals).abs() / (predictions.abs() + actuals.abs())).mean()


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
