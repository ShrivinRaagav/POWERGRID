import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from src.models.base_model import BaseForecastModel
from src.models.registry import register_model
from src.evaluation.metrics import evaluate_all_metrics
from src.utils.helpers import setup_logger

logger = setup_logger("random_forest")

@register_model("random_forest")
class RandomForestForecastModel(BaseForecastModel):
    """
    Random Forest Regressor forecasting model for material demand prediction.
    Obtains all hyperparameters from config.yaml.
    """
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 10, min_samples_split: int = 2, random_state: int = 42, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            **self.kwargs
        )
        self.feature_importances_ = None
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
        logger.info("Training RandomForestRegressor...")
        self.feature_cols = list(X_train.columns)
        
        # Calculate training medians for robust imputation
        self.impute_values = X_train.median().fillna(0.0)
        X_train_imputed = X_train.fillna(self.impute_values)
        
        self.model.fit(X_train_imputed, y_train)
        self.feature_importances_ = self.model.feature_importances_
        self.is_fitted = True
        logger.info("RandomForestRegressor training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RandomForestRegressor is not fitted yet.")
        # Ensure columns are aligned and imputed
        X_aligned = X[self.feature_cols]
        X_imputed = X_aligned.fillna(self.impute_values)
        return self.model.predict(X_imputed)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating RandomForestRegressor...")
        preds = self.predict(X)
        return evaluate_all_metrics(y, preds)

    def save(self, filepath: str) -> None:
        logger.info(f"Saving RandomForestRegressor to {filepath}...")
        state = {
            "model": self.model,
            "feature_importances": self.feature_importances_,
            "feature_cols": self.feature_cols,
            "impute_values": self.impute_values,
            "is_fitted": self.is_fitted,
            "hyperparams": self.get_hyperparameters()
        }
        # Save to runtime checkpoint
        joblib.dump(state, filepath)
        # Save copy to final artifacts path
        artifacts_path = "artifacts/models/random_forest.joblib"
        os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
        joblib.dump(state, artifacts_path)
        logger.info("RandomForestRegressor successfully saved.")

    def load(self, filepath: str) -> None:
        logger.info(f"Loading RandomForestRegressor from {filepath}...")
        state = joblib.load(filepath)
        self.model = state["model"]
        self.feature_importances_ = state["feature_importances"]
        self.feature_cols = state["feature_cols"]
        self.impute_values = state.get("impute_values")
        self.is_fitted = state["is_fitted"]
        # Sync attributes with loaded hyperparameters
        hparams = state["hyperparams"]
        self.n_estimators = hparams.get("n_estimators", 100)
        self.max_depth = hparams.get("max_depth", 10)
        self.min_samples_split = hparams.get("min_samples_split", 2)
        self.random_state = hparams.get("random_state", 42)

    def get_model_name(self) -> str:
        return "random_forest"

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            **self.kwargs
        }
