import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from xgboost import XGBRegressor
from src.models.base_model import BaseForecastModel
from src.models.registry import register_model
from src.evaluation.metrics import evaluate_all_metrics
from src.utils.helpers import setup_logger

logger = setup_logger("xgboost")

@register_model("xgboost")
class XGBoostForecastModel(BaseForecastModel):
    """
    XGBoost Regressor forecasting model for material demand prediction.
    Supports early stopping utilizing the validation set.
    """
    def __init__(self, learning_rate: float = 0.05, max_depth: int = 5, n_estimators: int = 150, subsample: float = 0.8, colsample_bytree: float = 0.8, early_stopping_rounds: Optional[int] = 10, random_state: int = 42, **kwargs):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.model = XGBRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
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
        logger.info("Training XGBoost Regressor...")
        self.feature_cols = list(X_train.columns)
        
        self.impute_values = X_train.median().fillna(0.0)
        X_train_imputed = X_train.fillna(self.impute_values)
        
        if self.early_stopping_rounds is not None and X_val is not None and y_val is not None:
            logger.info(f"Training with early stopping (rounds={self.early_stopping_rounds}) using Validation Set.")
            X_val_imputed = X_val.fillna(self.impute_values)
            self.model.fit(
                X_train_imputed, y_train,
                eval_set=[(X_val_imputed, y_val)],
                verbose=False
            )
            if hasattr(self.model, "best_iteration"):
                logger.info(f"XGBoost best iteration restored: {self.model.best_iteration}")
        else:
            if self.early_stopping_rounds is not None:
                logger.warning("Early stopping is enabled but validation set is missing. Fitting without early stopping.")
            self.model.fit(X_train_imputed, y_train, verbose=False)
            
        self.is_fitted = True
        logger.info("XGBoost Regressor training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("XGBoost Regressor is not fitted yet.")
        X_aligned = X[self.feature_cols]
        X_imputed = X_aligned.fillna(self.impute_values)
        return self.model.predict(X_imputed)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating XGBoost Regressor...")
        preds = self.predict(X)
        return evaluate_all_metrics(y, preds)

    def save(self, filepath: str) -> None:
        logger.info(f"Saving XGBoost Regressor to {filepath}...")
        state = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "impute_values": self.impute_values,
            "is_fitted": self.is_fitted,
            "hyperparams": self.get_hyperparameters()
        }
        # Save runtime checkpoint
        joblib.dump(state, filepath)
        # Save to artifacts directory
        artifacts_path = "artifacts/models/xgboost.joblib"
        os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
        joblib.dump(state, artifacts_path)
        logger.info("XGBoost Regressor successfully saved.")

    def load(self, filepath: str) -> None:
        logger.info(f"Loading XGBoost Regressor from {filepath}...")
        state = joblib.load(filepath)
        self.model = state["model"]
        self.feature_cols = state["feature_cols"]
        self.impute_values = state.get("impute_values")
        self.is_fitted = state["is_fitted"]
        # Sync attributes with loaded hyperparameters
        hparams = state["hyperparams"]
        self.learning_rate = hparams.get("learning_rate", 0.05)
        self.max_depth = hparams.get("max_depth", 5)
        self.n_estimators = hparams.get("n_estimators", 150)
        self.subsample = hparams.get("subsample", 0.8)
        self.colsample_bytree = hparams.get("colsample_bytree", 0.8)
        self.early_stopping_rounds = hparams.get("early_stopping_rounds", 10)
        self.random_state = hparams.get("random_state", 42)

    def get_model_name(self) -> str:
        return "xgboost"

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
            **self.kwargs
        }
