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
        model_params (dict): Parameters of the forecast model.
        model_type (string): Type of the model which can be either "python"
             or "r" depending on the modeling language.
        save_model (bool): When this is true, the trained model will be 
            saved as a file; otherwise it will not be saved. The default 
            value is False. 
    """
    def __init__(
        self, 
        df_config,
        model_params=None,
        model_type="python",
        save_model=False
    ):
        super().__init__(df_config)
        self.model_params = model_params
        self.model_type = model_type
        self.save_model = save_model

    @property
    def model_type(self):
        return self._model_type
    
    @model_type.setter
    def model_type(self, val):
        if val in ["python", "r"]:
            self._model_type = val
        else:
            raise Exception("Invalid model type is given. Please choose from \"python\" or \"r\!")

    #@abstractmethod
    def fit(self, X, y):
        """
        Fit a forecasting model.
        """
        return self

    #@abstractmethod
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

if __name__ == "__main__":
    df_config = {'time_col_name': 'timestamp', 'target_col_name': 'sales', 'frequency': 'MS', 'time_format': '%m/%d/%Y'}
    dummy_forecaster = BaseTSForecaster(df_config, model_type="r")