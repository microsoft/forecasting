def MAPE(predictions, actuals):
    """
    Implements Mean Absolute Prediction Error (MAPE).

    Args:
        predictions (pandas.Series): a vector of predicted values.
        actuals (pandas.Series): a vector of actual values.

    Returns:
        MAPE value
    """
    return ((predictions - actuals).abs() / actuals).mean()


def sMAPE(predictions, actuals):
    """
    Implements Symmetric Mean Absolute Prediction Error (sMAPE).

    Args:
        predictions (pandas.Series): a vector of predicted values.
        actuals (pandas.Series): a vector of actual values.

    Returns:
        sMAPE value
    """
    return ((predictions - actuals).abs() / (predictions.abs() + actuals.abs())).mean()