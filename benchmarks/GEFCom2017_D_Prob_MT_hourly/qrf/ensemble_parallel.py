# this file replaces quantile/ensemble.py file of scikit-garden package
# this code, unlike the original code, makes use of all available cores when doing scoring
# also, unlike the original code, predict() function in the new code can generate predictions for multiple quantiles

import numpy as np
from numpy import ma
from sklearn.ensemble.forest import ForestRegressor
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from joblib import Parallel, delayed
from skgarden.quantile.tree import DecisionTreeQuantileRegressor
from skgarden.quantile.ensemble import generate_sample_indices
from ensemble_parallel_utils import weighted_percentile_vectorized

class BaseForestQuantileRegressor(ForestRegressor):
    """Training and scoring of Quantile Regression Random Forest

    Training code is the same as in scikit-garden package. Scoring code uses all cores, unlike the original 
    code in scikit-garden.

    Attributes:
        y_train_ : array-like, shape=(n_samples,)
            Cache the target values at fit time.

        y_weights_ : array-like, shape=(n_estimators, n_samples)
            y_weights_[i, j] is the weight given to sample ``j` while
            estimator ``i`` is fit. If bootstrap is set to True, this
            reduces to a 2-D array of ones.

        y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
            y_train_leaves_[i, j] provides the leaf node that y_train_[i]
            ends up when estimator j is fit. If y_train_[i] is given
            a weight of zero when estimator j is fit, then the value is -1.

    """
    def fit(self, X, y):
        """Builds a forest from the training set (X, y).

        Args:
            X : array-like or sparse matrix, shape = [n_samples, n_features]
                The training input samples. Internally, it will be converted to
                ``dtype=np.float32`` and if a sparse matrix is provided
                to a sparse ``csc_matrix``.

            y : array-like, shape = [n_samples] or [n_samples, n_outputs]
                The target values (class labels) as integers or strings.

            sample_weight : array-like, shape = [n_samples] or None
                Sample weights. If None, then samples are equally weighted. Splits
                that would create child nodes with net zero or negative weight are
                ignored while searching for a split in each node. Splits are also
                ignored if they would result in any single class carrying a
                negative weight in either child node.

            check_input : boolean, (default=True)
                Allow to bypass several input checking.
                Don't use this parameter unless you know what you do.

            X_idx_sorted : array-like, shape = [n_samples, n_features], optional
                The indexes of the sorted training input samples. If many tree
                are grown on the same dataset, this allows the ordering to be
                cached between trees. If None, the data will be sorted here.
                Don't use this parameter unless you know what to do.

        Returns:
            self : object
                Returns self.
        """
        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(BaseForestQuantileRegressor, self).fit(X, y)

        self.y_train_ = y
        self.y_train_leaves_ = -np.ones((self.n_estimators, len(y)), dtype=np.int32)
        self.y_weights_ = np.zeros_like((self.y_train_leaves_), dtype=np.float32)

        for i, est in enumerate(self.estimators_):
            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, len(y))
            else:
                bootstrap_indices = np.arange(len(y))

            est_weights = np.bincount(bootstrap_indices, minlength=len(y))
            y_train_leaves = est.y_train_leaves_
            for curr_leaf in np.unique(y_train_leaves):
                y_ind = y_train_leaves == curr_leaf
                self.y_weights_[i, y_ind] = (
                    est_weights[y_ind] / np.sum(est_weights[y_ind]))

            self.y_train_leaves_[i, bootstrap_indices] = y_train_leaves[bootstrap_indices]

        return self

    def _compute_percentiles(self, x_leaf, quantiles, sorter):
        mask = self.y_train_leaves_ != np.expand_dims(x_leaf, 1)
        x_weights = ma.masked_array(self.y_weights_, mask)
        weights = x_weights.sum(axis=0)
        return weighted_percentile_vectorized(self.y_train_, quantiles, weights, sorter)

    def predict(self, X, quantiles=None):
        """Predict regression value for X.

        Args:
            X : array-like or sparse matrix of shape = [n_samples, n_features]
                The input samples. Internally, it will be converted to
                ``dtype=np.float32`` and if a sparse matrix is provided
                to a sparse ``csr_matrix``.

            quantiles : list of ints, optional
                list of value ranging from 0 to 100. By default, the mean is returned.

            check_input : boolean, (default=True)
                Allow to bypass several input checking.
                Don't use this parameter unless you know what you do.

        Returns:
            y : array of shape = [n_samples]
                If quantile is set to None, then return E(Y | X). Else return
                y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantiles is None:
            return super(BaseForestQuantileRegressor, self).predict(X)

        sorter = np.argsort(self.y_train_)
        X_leaves = self.apply(X)

        with Parallel(n_jobs=-1, backend="multiprocessing", batch_size=10) as p:
            percentiles = p(delayed(self._compute_percentiles)(x_leaf, quantiles, sorter) for x_leaf in X_leaves)
            return np.array(percentiles)


class RandomForestQuantileRegressor(BaseForestQuantileRegressor):
    """A random forest regressor that provides quantile estimates.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    References:
        Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf

    Attributes:
        estimators_ : list of DecisionTreeQuantileRegressor
            The collection of fitted sub-estimators.

        feature_importances_ : array of shape = [n_features]
            The feature importances (the higher, the more important the feature).

        n_features_ : int
            The number of features when ``fit`` is performed.

        n_outputs_ : int
            The number of outputs when ``fit`` is performed.

        oob_score_ : float
            Score of the training dataset obtained using an out-of-bag estimate.

        oob_prediction_ : array of shape = [n_samples]
            Prediction computed with out-of-bag estimate on the training set.
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        """Initialize RandomForestQuantileRegressor class

        Args:
            n_estimators : integer, optional (default=10)
                The number of trees in the forest.

            criterion : string, optional (default="mse")
                The function to measure the quality of a split. Supported criteria
                are "mse" for the mean squared error, which is equal to variance
                reduction as feature selection criterion, and "mae" for the mean
                absolute error.
                .. versionadded:: 0.18
                    Mean Absolute Error (MAE) criterion.

            max_features : int, float, string or None, optional (default="auto")
                The number of features to consider when looking for the best split:
                - If int, then consider `max_features` features at each split.
                - If float, then `max_features` is a percentage and
                    `int(max_features * n_features)` features are considered at each
                    split.
                - If "auto", then `max_features=n_features`.
                - If "sqrt", then `max_features=sqrt(n_features)`.
                - If "log2", then `max_features=log2(n_features)`.
                - If None, then `max_features=n_features`.
                Note: the search for a split does not stop until at least one
                valid partition of the node samples is found, even if it requires to
                effectively inspect more than ``max_features`` features.

            max_depth : integer or None, optional (default=None)
                The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure or until all leaves contain less than
                min_samples_split samples.

            min_samples_split : int, float, optional (default=2)
                The minimum number of samples required to split an internal node:
                - If int, then consider `min_samples_split` as the minimum number.
                - If float, then `min_samples_split` is a percentage and
                    `ceil(min_samples_split * n_samples)` are the minimum
                    number of samples for each split.
                .. versionchanged:: 0.18
                    Added float values for percentages.

            min_samples_leaf : int, float, optional (default=1)
                The minimum number of samples required to be at a leaf node:
                - If int, then consider `min_samples_leaf` as the minimum number.
                - If float, then `min_samples_leaf` is a percentage and
                    `ceil(min_samples_leaf * n_samples)` are the minimum
                    number of samples for each node.
                .. versionchanged:: 0.18
                    Added float values for percentages.

            min_weight_fraction_leaf : float, optional (default=0.)
                The minimum weighted fraction of the sum total of weights (of all
                the input samples) required to be at a leaf node. Samples have
                equal weight when sample_weight is not provided.

            max_leaf_nodes : int or None, optional (default=None)
                Grow trees with ``max_leaf_nodes`` in best-first fashion.
                Best nodes are defined as relative reduction in impurity.
                If None then unlimited number of leaf nodes.

            bootstrap : boolean, optional (default=True)
                Whether bootstrap samples are used when building trees.

            oob_score : bool, optional (default=False)
                whether to use out-of-bag samples to estimate
                the R^2 on unseen data.

            n_jobs : integer, optional (default=1)
                The number of jobs to run in parallel for both `fit` and `predict`.
                If -1, then the number of jobs is set to the number of cores.

            random_state : int, RandomState instance or None, optional (default=None)
                If int, random_state is the seed used by the random number generator;
                If RandomState instance, random_state is the random number generator;
                If None, the random number generator is the RandomState instance used
                by `np.random`.

            verbose : int, optional (default=0)
                Controls the verbosity of the tree building process.

            warm_start : bool, optional (default=False)
                When set to ``True``, reuse the solution of the previous call to fit
                and add more estimators to the ensemble, otherwise, just fit a whole
                new forest.
        """
        super(RandomForestQuantileRegressor, self).__init__(
            base_estimator=DecisionTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


