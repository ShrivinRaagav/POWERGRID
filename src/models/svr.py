import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.svm import SVR
from src.models.base_model import BaseForecastModel
from src.models.registry import register_model
from src.evaluation.metrics import evaluate_all_metrics
from src.utils.helpers import setup_logger

logger = setup_logger("svr")

@register_model("svr")
class SVRForecastModel(BaseForecastModel):
    """
    Support Vector Regression (SVR) forecasting model for material demand prediction.
    Obtains kernel, C, gamma, and epsilon from config.yaml.
    """
    def __init__(self, kernel: str = "rbf", C: float = 10.0, epsilon: float = 0.1, gamma: str = "scale", **kwargs):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.kwargs = kwargs
        
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            **self.kwargs
        )
        self.feature_cols = None
        self.impute_values = None
        self.is_fitted = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        logger.info("Training SVR...")
        self.feature_cols = list(X_train.columns)
        
        # Calculate training medians for robust imputation
        self.impute_values = X_train.median().fillna(0.0)
        X_train_imputed = X_train.fillna(self.impute_values)
        
        self.model.fit(X_train_imputed, y_train)
        self.is_fitted = True
        logger.info("SVR training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SVR is not fitted yet.")
        X_aligned = X[self.feature_cols]
        X_imputed = X_aligned.fillna(self.impute_values)
        return self.model.predict(X_imputed)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating SVR...")
        preds = self.predict(X)
        return evaluate_all_metrics(y, preds)

    def save(self, filepath: str) -> None:
        logger.info(f"Saving SVR to {filepath}...")
        state = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "impute_values": self.impute_values,
            "is_fitted": self.is_fitted,
            "hyperparams": self.get_hyperparameters()
        }
        # Save to runtime checkpoint
        joblib.dump(state, filepath)
        # Save copy to final artifacts path
        artifacts_path = "artifacts/models/svr.joblib"
        os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
        joblib.dump(state, artifacts_path)
        logger.info("SVR successfully saved.")

    def load(self, filepath: str) -> None:
        logger.info(f"Loading SVR from {filepath}...")
        state = joblib.load(filepath)
        self.model = state["model"]
        self.feature_cols = state["feature_cols"]
        self.impute_values = state.get("impute_values")
        self.is_fitted = state["is_fitted"]
        # Sync attributes with loaded hyperparameters
        hparams = state["hyperparams"]
        self.kernel = hparams.get("kernel", "rbf")
        self.C = hparams.get("C", 10.0)
        self.epsilon = hparams.get("epsilon", 0.1)
        self.gamma = hparams.get("gamma", "scale")

    def get_model_name(self) -> str:
        return "svr"

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            **self.kwargs
        }
