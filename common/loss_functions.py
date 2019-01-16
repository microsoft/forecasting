import pandas as pd

def pinball_loss(predictions, actuals, q):
    """
    Implements pinball loss evaluation function.

    Args:
        predictions (pandas.Series): a vector of predicted values.
        actuals (pandas.Series): a vector of actual values.
        q (pandas.Series): a vector of quantiles.

    Returns:
        A pandas Series of pinball loss values for each prediction.
    """
    zeros = pd.Series([0]*len(predictions))
    return (predictions-actuals).combine(zeros, max)*(1-q) + (actuals-predictions).combine(zeros, max)*q