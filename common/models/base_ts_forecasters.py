"""
Base class for time series forecasting models.
"""
import pickle
from abc import abstractmethod
from sklearn.base import RegressorMixin
from common.features.base_ts_estimators import BaseTSEstimator

class BaseTSForecaster(BaseTSEstimator, RegressorMixin):
    """
    Base abstract forecaster class for all time series forecasting models. 

    Args:
        df_config (dict): Configuration of the time series data frame used 
            for building the forecast model.
        model_hparams (dict): Hyperparameters of the forecast model.
        model_type (string): Type of the model which can be either "python"
             or "r" depending on the modeling language.

    Attributes: 
        model (object): Object of the forecasting model.
    """
    def __init__(
        self, 
        df_config,
        model_hparams=None,
        model_type="python"
    ):
        super().__init__(df_config)
        self.model_hparams = model_hparams
        self.model_type = model_type
        self.model=None

    @property
    def model_type(self):
        return self._model_type
    
    @model_type.setter
    def model_type(self, val):
        if val in ["python", "r"]:
            self._model_type = val
        else:
            raise Exception("Invalid model type is given. Please choose from \"python\" or \"r\!")

    def save(self, file_name):
        """
        Save the model.
        """
        try:
            pickle.dump(self.model, open(file_name, "wb"))
        except IOError as e:
            print(e)

    def load(self, file_name):
        """
        Load a trained model.
        """
        try:
            self.model = pickle.load(open(file_name, "rb"))
        except IOError as e:
            print(e)  

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

    @abstractmethod
    def eval(self, X, y):
        """
        Compute the forecasting accuracy.
        """
        return self


# if __name__ == "__main__":
#     df_config = {'time_col_name': 'timestamp', 'target_col_name': 'sales', 'frequency': 'MS', 'time_format': '%m/%d/%Y'}
#     dummy_forecaster = BaseTSForecaster(df_config, model_type="r")
#     dummy_forecaster.save("./dummpy_model.pkl")
#     dummy_forecaster.load("./dummpy_model.pkl")