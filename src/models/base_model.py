from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union, Optional
import pandas as pd
import numpy as np

class BaseForecastModel(ABC):
    """
    Abstract Base Class for all machine learning forecasting models.
    All future models (e.g. LSTM, XGBoost, Random Forest) must inherit from 
    this class and implement its abstract methods.
    """
    
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        """
        Trains the forecasting model on the training dataset.
        
        Parameters:
        X_train (pd.DataFrame): Training feature columns.
        y_train (pd.Series): Training target column (Quantity_Required).
        X_val (pd.DataFrame): Optional validation feature columns.
        y_val (pd.Series): Optional validation target column.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates point predictions (forecasts) for input features.
        
        Parameters:
        X (pd.DataFrame): Input features.
        
        Returns:
        np.ndarray: 1D array of predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluates the model on a test/validation dataset and returns a dictionary 
        of metrics (MAE, RMSE, MAPE, SMAPE, R2).
        
        Parameters:
        X (pd.DataFrame): Evaluation features.
        y (pd.Series): Actual target values.
        
        Returns:
        Dict[str, float]: Dictionary containing computed forecasting metrics.
        """
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """
        Serializes the model state to disk.
        
        Parameters:
        filepath (str): Path to write the serialized model file.
        """
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        """
        Loads the model state from disk.
        
        Parameters:
        filepath (str): Path to the serialized model file.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Returns the human-readable identifier of the model.
        
        Returns:
        str: Model identifier.
        """
        pass

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Returns the hyperparameters used to configure the model.
        
        Returns:
        Dict[str, Any]: Dictionary of configuration hyperparameters.
        """
        pass

import typing
Optional = typing.Optional
