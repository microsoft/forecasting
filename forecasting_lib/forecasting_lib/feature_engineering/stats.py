# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from forecasting_lib.feature_engineering.base_ts_estimators import BaseTSFeaturizer
from forecasting_lib.feature_engineering.utils import is_iterable_but_not_string


class PopularityFeaturizer(BaseTSFeaturizer):
    """
    Computes a feature indicating the popularity of each group of data.

    Args:
        df_config(dict): Configuration of the time series data frame
            to compute features on.
        id_col_name(str): Name of the column identifying different groups to
            compare popularity on. For example, if you want to compare the
            competitiveness of the prices of different juice brands,
            the id_col_name is the name of the brand column.
        feature_col_name(str): Name of the column used to measure popularity,
            e.g. prices of different juice brands can indicate competitiveness.
            If data_format is 'wide', all the wide_col_names should start
            with feature_col_name and feature_col_name column doesn't have to
            exist in the data frame.
        data_format(str): How the columns for measuring popularity are
            organized. Accepted values are 'long' and 'wide'.
            If 'long', only the feature_col_name column is used to compute
            the popularity.
            If 'wide', wide_col_names must be provided and are used to
            compute the popularity.
            It's a read-only property and can not be changed once a
            featurizer is instantiated.
        wide_col_names(list of str): If data_format is 'wide', each unique
            value in the id_col_name column has a corresponding wide column
            that contains the feature value for that unique id. The column
            names of the wide columns should be composed of the
            feature_col_name and id. For example, if the feature_col_name is
            'price' and there are three brands, 1, 2, 3, the wide_col_names
            should be ['price1', 'price2', 'price3'].
            wide_col_names must be set when data_format is 'wide'.
        output_col_name(str): Name of the output feature column. Default
            value is 'popularity'.
        return_feature_col(bool): When data_format is 'wide', if the feature
            column in the long format should be included in the output. For
            example, if wide_col_names is ['price1', 'price2', 'price3'] and
            return_feature_col is True, a 'price' column will be extracted
            from the wide columns added to the output. Default value is False.

    Examples:
        import pandas as pd
        >>>df_config = {
        ...    'time_col_name': 'date',
        ...    'ts_id_col_names': 'brand',
        ...    'target_col_name': 'sales',
        ...    'frequency': 'D',
        ...    'time_format': '%Y-%m-%d'
        ...}
        >>>tsdf_long = pd.DataFrame({
        ...    'brand': [1] * 2 + [2] * 2 + [3] * 2,
        ...    'date': list(pd.date_range('2011-01-01', '2011-01-02')) * 3,
        ...    'price': [10, 11, 9, 12, 9, 8],
        ...    'sales': [1, 2, 3, 4, 5, 6]})
        >>>popularity_featurizer_long = PopularityFeaturizer(
        ...   df_config,
        ...   id_col_name='brand',
        ...   feature_col_name='price',
        ...   data_format='long')
        >>> popularity_featurizer_long.transform(tsdf_long)
           brand       date  price  sales  popularity
        0      1 2011-01-01     10      1    1.071429
        1      1 2011-01-02     11      2    1.064516
        2      2 2011-01-01      9      3    0.964286
        3      2 2011-01-02     12      4    1.161290
        4      3 2011-01-01      9      5    0.964286
        5      3 2011-01-02      8      6    0.774194

        >>>tsdf_wide = pd.DataFrame({
        ...    'brand': [1] * 2 + [2] * 2 + [3] * 2,
        ...    'date': list(pd.date_range('2011-01-01', '2011-01-02')) * 3,
        ...    'price1': [10, 11, 10, 11, 10, 11],
        ...    'price2': [9, 12, 9, 12, 9, 12],
        ...    'price3': [9, 8, 9, 8, 9, 8],
        ...    'sales': [1, 2, 3, 4, 5, 6]})

        >>>popularity_featurizer_wide = PopularityFeaturizer(
        ...    df_config,
        ...    id_col_name='brand',
        ...    feature_col_name='price',
        ...    data_format='wide',
        ...    wide_col_names=['price1', 'price2', 'price3'],
        ...    return_feature_col=True
        ...)

        >>>popularity_featurizer_wide.transform(tsdf_wide)
           brand       date  price1  price2  price3  sales  popularity  price
        0      1 2011-01-01      10       9       9      1    1.071429     10
        1      1 2011-01-02      11      12       8      2    1.064516     11
        2      2 2011-01-01      10       9       9      3    0.964286      9
        3      2 2011-01-02      11      12       8      4    1.161290     12
        4      3 2011-01-01      10       9       9      5    0.964286      9
        5      3 2011-01-02      11      12       8      6    0.774194      8
    """

    def __init__(
        self,
        df_config,
        id_col_name,
        feature_col_name,
        data_format="long",
        wide_col_names=None,
        output_col_name="popularity",
        return_feature_col=False,
    ):

        super().__init__(df_config)
        self.id_col_name = id_col_name
        self._data_format = data_format
        self.feature_col_name = feature_col_name
        self.wide_col_names = wide_col_names
        self.output_col_name = output_col_name
        self.return_feature_col = return_feature_col

        if data_format not in ["long", "wide"]:
            raise ValueError(
                "Invalid value for argument data_format, " "accepted values are {0} and {1}".format("long", "wide")
            )

    @property
    def data_format(self):
        return self._data_format

    @property
    def wide_col_names(self):
        return self._wide_col_names

    @wide_col_names.setter
    def wide_col_names(self, val):
        if self.data_format == "wide" and val is None:
            raise ValueError("For wide data format, wide_col_names can not be " "None.")
        if self.data_format == "wide":
            if is_iterable_but_not_string(val):
                val = list(val)
            else:
                raise ValueError("wide_col_names must be a non-string Iterable, " "e.g. a list.")
            for c in val:
                if not c.startswith(self.feature_col_name):
                    raise ValueError("Elements of wide_col_names must start " "with feature_col_name")

        self._wide_col_names = val

    def fit(self, X, y=None):
        """
        To be compatible with scikit-learn interface. Nothing needs to be
        done at the fit stage for this featurizer.
        """
        return self

    def _add_wide_cols(self, df, input_col):
        id_all = df[self.id_col_name].unique()
        for id in id_all:
            wide_col_name = input_col + str(id)
            df[wide_col_name] = df.loc[df[self.id_col_name] == id, input_col].values[0]
        return df

    def transform(self, X):
        self._check_config_cols_exist(X)
        if self.data_format == "long":
            if len(self.ts_id_col_names) > 0:
                grain_col_name_tmp = self.ts_id_col_names.copy()
                if self.id_col_name in self.ts_id_col_names:
                    grain_col_name_tmp.remove(self.id_col_name)
                cols = grain_col_name_tmp + [
                    self.time_col_name,
                    self.feature_col_name,
                    self.id_col_name,
                ]
                group_cols = grain_col_name_tmp + [self.time_col_name]
            else:
                cols = [
                    self.time_col_name,
                    self.feature_col_name,
                    self.id_col_name,
                ]
                group_cols = self.time_col_name
            X_tmp = X[cols].copy()
            id_all = X_tmp[self.id_col_name].unique()
            self.wide_col_names = []
            for id in id_all:
                wide_col_name = self.feature_col_name + str(id)
                # X_tmp[wide_col_name] = np.nan
                self.wide_col_names.append(wide_col_name)

            X_tmp = X_tmp.groupby(group_cols).apply(lambda g: self._add_wide_cols(g, self.feature_col_name))
            X_tmp.reset_index(inplace=True)

        else:
            X_tmp = X[self.wide_col_names + [self.id_col_name]].copy()
            X_tmp[self.feature_col_name] = X_tmp.apply(
                lambda x: x.loc[self.feature_col_name + str(int(x.loc[self.id_col_name]))], axis=1,
            )

        X_tmp["avg"] = X_tmp[self.wide_col_names].sum(axis=1).apply(lambda x: x / len(self.wide_col_names))
        X[self.output_col_name] = X_tmp[self.feature_col_name] / X_tmp["avg"]

        if self.return_feature_col:
            X[self.feature_col_name] = X_tmp[self.feature_col_name]

        return X
