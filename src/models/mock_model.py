import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional
from src.models.base_model import BaseForecastModel
from src.models.registry import register_model
from src.evaluation.metrics import evaluate_all_metrics
from src.utils.helpers import setup_logger

logger = setup_logger("mock_model")

@register_model("mock_model")
class MockForecastModel(BaseForecastModel):
    """
    A simple forecasting model to test the ML infrastructure interfaces and CLI controller.
    It returns a constant prediction with Gaussian noise.
    """
    def __init__(self, constant_value: float = 0.0, random_noise_std: float = 0.1, **kwargs):
        self.constant_value = constant_value
        self.random_noise_std = random_noise_std
        self.fitted_mean = constant_value
        self.is_fitted = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        logger.info("Training MockForecastModel (computing target mean)...")
        # Simulating fitting: save training target mean as base prediction
        self.fitted_mean = float(y_train.mean())
        self.is_fitted = True
        logger.info(f"MockForecastModel trained. Fitted Mean target: {self.fitted_mean:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            logger.warning("MockForecastModel not fitted yet. Predicting default constant_value.")
            base = self.constant_value
        else:
            base = self.fitted_mean
            
        np.random.seed(42)  # For deterministic evaluation in tests
        noise = np.random.normal(0, self.random_noise_std, size=len(X))
        return np.maximum(0.0, base + noise) # Demand cannot be negative

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating MockForecastModel...")
        preds = self.predict(X)
        return evaluate_all_metrics(y, preds)

    def save(self, filepath: str) -> None:
        logger.info(f"Saving MockForecastModel parameters to {filepath}...")
        state = {
            "constant_value": self.constant_value,
            "random_noise_std": self.random_noise_std,
            "fitted_mean": self.fitted_mean,
            "is_fitted": self.is_fitted
        }
        joblib.dump(state, filepath)

    def load(self, filepath: str) -> None:
        logger.info(f"Loading MockForecastModel parameters from {filepath}...")
        state = joblib.load(filepath)
        self.constant_value = state["constant_value"]
        self.random_noise_std = state["random_noise_std"]
        self.fitted_mean = state["fitted_mean"]
        self.is_fitted = state["is_fitted"]

    def get_model_name(self) -> str:
        return "mock_model"

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "constant_value": self.constant_value,
            "random_noise_std": self.random_noise_std
        }
