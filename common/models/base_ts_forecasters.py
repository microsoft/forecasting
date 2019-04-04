"""
Base class for time series forecasting models.
"""
from abc import abstractmethod
from sklearn.base import RegressorMixin
from common.features.base_ts_estimators import BaseTSEstimator

class BaseTSForecaster(BaseTSEstimator, RegressorMixin):
    """
    Base abstract forecaster class for all time series forecasting models. 

    Args:
        df_config (dict): Configuration of the time series data frame used 
            for building the forecast model.
        fit_pred_config (dict): Configuration of the model fitting and 
            prediction including the range of training data and the range
            of testing data in each forecast round.
        model_params (dict): Parameters of the forecast model.
        save_model (bool): When this is true, the trained model will be 
            saved as a file; otherwise it will not be saved. The default 
            value is False. 
    """
    def __init__(
        self, 
        df_config
        fit_pred_config,
        model_params=None,
        save_model=False
    ):
        super().__init__(df_config)
        self.fit_pred_config = fit_pred_config
        self.model_params
        self.save_model

    @abstractmethod
    def fit(self, X, y):
        """
        Fit a forecasting model.
        """
        return self

    @abstractmethod
    def predict(self, X):
        """
        Predict using the forecasting model.

        Args:
            X (pandas.DataFrame) : Input data frame of the features with
                shape (n_samples, n_features).

        Returns:
            pandas.DataFrame: Output data frame containing the predicted
                values.
        """
        return X