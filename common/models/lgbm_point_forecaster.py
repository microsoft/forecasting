import pandas as pd
import lightgbm as lgb
from base_ts_forecasters import BaseTSForecaster
from common.evaluation_utils import MAPE

class LGBMPointForecaster(BaseTSForecaster):
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

    def fit(self, X, y):
        # Create training set
        dtrain = lgb.Dataset(X, label = y)
        self.model = lgb.train(
            params, 
            dtrain, 
            valid_sets = [dtrain], 
            #categorical_feature = categ_fea,
            verbose_eval =  False
        )

    def predict(
            self, 
            X,
            apply_round=False):
        predictions = pd.DataFrame({self.df_config["target_col_name"]: self.model.predict(X)})
        if apply_round: 
            predictions[self.df_config["target_col_name"]] = predictions[self.df_config["target_col_name"]].apply(lambda x: round(x))
        self.predictions = pd.concat([df[self.extra_pred_col_names].reset_index(drop=True), predictions], axis=1)
        return self.predictions

    def eval(self, y, actual_col_name="actual"):
        if self.predictions is None:
            raise Exception("No valid prediction results are found for evaluation! " 
                            "Please call the fit() method to generate forecasts first.")
        else:
            combined = pd.merge(self.predictions, y, on=self.extra_pred_col_names, how='left')
            return MAPE(combined[self.df_config["target_col_name"]], result[actual_col_name])*100


if __name__ == "__main__":
    df_config = {'time_col_name': 'timestamp', 'target_col_name': 'sales', 'frequency': 'MS', 'time_format': '%m/%d/%Y'}
    submission_config = {'time_col_name': 'week'}
    LGBM_forecaster = LGBMPointForecaster(df_config, submission_config)
    print("A LGBM-point forecaster is created")