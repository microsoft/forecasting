import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from base_ts_forecasters import BaseTSForecaster
from common.evaluation_utils import MAPE

class LGBMForecaster(BaseTSForecaster):
    """
    Class for point forecast model using LightGBM package.
    """
    def __init__(
        self, 
        df_config,
        submission_config,
        model_hparams=None,
        model_type="python",
    ):
        super().__init__(df_config, model_hparams, model_type)
        self.submission_config = submission_config
        self.extra_pred_col_names = self.ts_id_col_names + [self.submission_config["time_col_name"]]
        self.predictions = None

        print(self.extra_pred_col_names)

    def fit(self, X, y, valid_size=0, train_mode=[]):
        """
        Fit a single model or multiple models.

        Args:
            train_mode (list): each element takes a value of either "train" or "skip" 
            indicating whether to train a model in this round or skip the training.
        """
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            self.model, _ = self._train_single_model(X, y, valid_size)
        elif isinstance(X, list) and isinstance(y, list) and isinstance(train_mode, list):
            assert len(X) == len(y) == len(train_mode)
            self.model = []
            for cur_X, cur_y, cur_mode in zip(X, y, train_mode):
                if cur_mode == "train":
                    cur_model, _ = self._train_single_model(cur_X, cur_y, valid_size)
                elif cur_mode == "skip":
                    cur_model = self.model[-1]
                else:
                    raise Exception("Invalid element found in train_mode!")
                self.model.append(cur_model)
        else:
            raise Exception("Invalid types of the input features and labels!")

    def predict(
            self, 
            X,
            apply_round=False, 
            combine_forecasts=True,
            forecast_round_idx=None):
        if isinstance(X, pd.DataFrame):
            self.predictions = self._predict_single_model(self.model, X, apply_round=True)
        elif isinstance(X, list):
            self.predictions = []
            for i, element in enumerate(zip(self.model, X)):
                cur_model, cur_X = element[0], element[1]
                cur_predictions = self._predict_single_model(cur_model, cur_X, apply_round=True)
                if forecast_round_idx is None:
                    cur_predictions["round"] = i+1
                else:
                    cur_predictions["round_idx"] = forecast_round_idx[i]
                self.predictions.append(cur_predictions)
            if combine_forecasts:
                self.predictions = pd.concat(self.predictions, axis=0)
        return self.predictions

    def _random_data_split(self, X, y, valid_size, random_state=1):
        """
        Randomly split the features and labels into training and validation sets.
        """
        train_feat, valid_feat, train_label, valid_label = train_test_split(X, y, test_size=valid_size, random_state=random_state)
        dtrain = lgb.Dataset(train_feat, train_label)
        dvalid = lgb.Dataset(valid_feat, valid_label)
        return dtrain, dvalid

    def _train_single_model(self, X, y, valid_size=0):
        from copy import deepcopy
        # Create training/validation sets
        if valid_size == 0:
            dtrain = lgb.Dataset(X, label = y)
            valid_sets = [dtrain]
        else:
            dtrain, dvalid = self._random_data_split(X, y, valid_size)
            valid_sets = [dtrain, dvalid]
        # Train model
        evals_result = {}
        model_hparams = deepcopy(self.model_hparams)
        categ_features = model_hparams["categorical_feature"]
        model_hparams.pop("categorical_feature")
        model = lgb.train(
            model_hparams, 
            dtrain, 
            valid_sets = valid_sets, 
            verbose_eval =  False,
            categorical_feature = categ_features,
            evals_result = evals_result
        )
        return model, evals_result

    def _predict_single_model(self, model, X, apply_round=False):
        predictions = pd.DataFrame({self.target_col_name: model.predict(X)})
        if apply_round: 
            predictions[self.target_col_name] = predictions[self.target_col_name].apply(lambda x: round(x))
        predictions = pd.concat([X[self.extra_pred_col_names].reset_index(drop=True), predictions], axis=1)
        predictions = predictions.sort_values(by=self.extra_pred_col_names).reset_index(drop=True)
        return predictions

if __name__ == "__main__":
    import os, sys, warnings
    import numpy as np
    import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_paths as bp
    import retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.benchmark_settings as bs
    from retail_sales.OrangeJuice_Pt_3Weeks_Weekly.submissions.LightGBM.make_features_new import make_features
    from retail_sales.OrangeJuice_Pt_3Weeks_Weekly.common.submission_utils import create_submission
    warnings.filterwarnings("ignore")

    df_config = {"time_col_name": "timestamp", "target_col_name": "move", "frequency": "MS", "time_format": "%m/%d/%Y", "ts_id_col_names": ["store", "brand"]}
    submission_config = {"time_col_name": "week"}
    feat_hparams = {"max_lag": 19, "window_size": 40}
    model_hparams = {"objective": "mape", "num_leaves": 124, "min_data_in_leaf": 340, "learning_rate": 0.1, 
                     "feature_fraction": 0.65, "bagging_fraction": 0.87, "bagging_freq": 19, "num_rounds": 940,
                     "early_stopping_rounds": 125, "num_threads": 4, "seed": 1}

    # Lags and categorical features
    lags = np.arange(2, feat_hparams["max_lag"]+1)
    used_columns = ["store", "brand", "week", "week_of_month", "month", "deal", "feat", "move", "price", "price_ratio"]
    categ_features = ["store", "brand", "deal"] 
    model_hparams["categorical_feature"] = categ_features
    print(model_hparams)

    # Model training and prediction 
    features_list = []
    labels_list = []
    test_feat_list = []
    train_mode = ["train", "skip"] #["train", "skip", "skip"]*4
    for r in range(2): #range(bs.NUM_ROUNDS):
        # Create features
        features = make_features(r, bp.TRAIN_DIR, lags, feat_hparams["window_size"], 0, used_columns, bs.store_list, bs.brand_list)
        print(features.head())
        train_feat = features[features.week <= bs.TRAIN_END_WEEK_LIST[r]].reset_index(drop=True)
        # Drop rows with NaN values
        train_feat.dropna(inplace=True)
        features_list.append(train_feat.drop("move", axis=1, inplace=False))
        labels_list.append(train_feat["move"])
        test_feat_list.append(features[features.week >= bs.TEST_START_WEEK_LIST[r]].reset_index(drop=True).drop("move", axis=1))

    LGBM_forecaster = LGBMForecaster(df_config, submission_config, model_hparams)
    print("A LGBM-point forecaster is created")

    LGBM_forecaster.fit(features_list, labels_list, train_mode=train_mode)

    print("Making predictions...") 
    LGBM_forecaster.predict(test_feat_list)

    print(LGBM_forecaster.predictions)

    # Generate submission
    raw_predictions = LGBM_forecaster.predictions
    #create_submission(raw_predictions, 0, "LightGBM")



    