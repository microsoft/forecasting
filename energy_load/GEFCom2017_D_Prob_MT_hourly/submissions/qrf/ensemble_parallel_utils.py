# this file replaces quantile/utils.py file in scikit-garden package
# the new vector_percentile_vetorized() function, unlike the original weighted_percentile() function, can compute percentiles for multiple quantiles

import numpy as np

def weighted_percentile_vectorized(a, quantiles, weights=None, sorter=None):
    """
    Returns the weighted percentile of a at q given weights.

    Parameters
    ----------
    a: array-like, shape=(n_samples,)
        samples at which the quantile.

    quantiles: array of ints
        list of quantiles.

    weights: array-like, shape=(n_samples,)
        weights[i] is the weight given to point a[i] while computing the
        quantile. If weights[i] is zero, a[i] is simply ignored during the
        percentile computation.

    sorter: array-like, shape=(n_samples,)
        If provided, assume that a[sorter] is sorted.

    Returns
    -------
    percentiles: array of floats
        Weighted percentile of a at each of quantiles.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method

    Notes
    -----
    Note that weighted_percentile(a, q) is not equivalent to
    np.percentile(a, q). This is because in np.percentile
    sorted(a)[i] is assumed to be at quantile 0.0, while here we assume
    sorted(a)[i] is given a weight of 1.0 / len(a), hence it is at the
    1.0 / len(a)th quantile.
    """
    if weights is None:
        weights = np.ones_like(a)

    a = np.asarray(a, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(a) != len(weights):
        raise ValueError("a and weights should have the same length.")

    if sorter is not None:
        a = a[sorter]
        weights = weights[sorter]

    nz = weights != 0
    a = a[nz]
    weights = weights[nz]

    if sorter is None:
        sorted_indices = np.argsort(a)
        sorted_a = a[sorted_indices]
        sorted_weights = weights[sorted_indices]
    else:
        sorted_a = a
        sorted_weights = weights

    # Step 1
    sorted_cum_weights = np.cumsum(sorted_weights)
    total = sorted_cum_weights[-1]

    # Step 2
    partial_sum = 100.0 / total * (sorted_cum_weights - sorted_weights / 2.0)

    percentiles = np.zeros_like(quantiles)
    for i, q in enumerate(quantiles):
        if q > 100 or q < 0:
            raise ValueError("q should be in-between 0 and 100, "
                             "got %d" % q)

        start = np.searchsorted(partial_sum, q) - 1
        if start == len(sorted_cum_weights) - 1:
            percentiles[i] = sorted_a[-1]
        elif start == -1:
            percentiles[i] = sorted_a[0]
        else:
            # Step 3.
            fraction = (q - partial_sum[start]) / (partial_sum[start + 1] - partial_sum[start])
            percentiles[i] = sorted_a[start] + fraction * (sorted_a[start + 1] - sorted_a[start])

    return percentiles
