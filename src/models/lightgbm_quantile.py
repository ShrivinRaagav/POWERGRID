import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from lightgbm import LGBMRegressor
from src.models.base_model import BaseForecastModel
from src.models.registry import register_model
from src.evaluation.metrics import evaluate_all_metrics
from src.utils.helpers import setup_logger

logger = setup_logger("lightgbm_quantile")

@register_model("lightgbm_quantile")
class LightGBMQuantileForecastModel(BaseForecastModel):
    """
    LightGBM Quantile Regression forecasting model.
    Trains three independent models corresponding to P10, P50, and P90 quantiles.
    """
    def __init__(self, learning_rate: float = 0.05, n_estimators: int = 100, num_leaves: int = 31, random_state: int = 42, **kwargs):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.kwargs = kwargs
        
        self.quantiles = [0.1, 0.5, 0.9]
        self.models = {}
        for q in self.quantiles:
            self.models[q] = LGBMRegressor(
                objective="quantile",
                alpha=q,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                num_leaves=self.num_leaves,
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
        logger.info("Training LightGBM Quantile Regressors (P10, P50, P90)...")
        self.feature_cols = list(X_train.columns)
        
        # Calculate training medians for robust imputation
        self.impute_values = X_train.median().fillna(0.0)
        X_train_imputed = X_train.fillna(self.impute_values)
        
        for q in self.quantiles:
            logger.info(f"Training LightGBM model for alpha = {q}...")
            if X_val is not None and y_val is not None:
                X_val_imputed = X_val.fillna(self.impute_values)
                self.models[q].fit(
                    X_train_imputed, y_train,
                    eval_set=[(X_val_imputed, y_val)],
                    callbacks=[]
                )
            else:
                self.models[q].fit(X_train_imputed, y_train)
                
        self.is_fitted = True
        logger.info("LightGBM Quantile Regressors training completed.")

    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("LightGBM Quantile Regressors are not fitted yet.")
            
        X_aligned = X[self.feature_cols]
        X_imputed = X_aligned.fillna(self.impute_values)
        p10 = self.models[0.1].predict(X_imputed)
        p50 = self.models[0.5].predict(X_imputed)
        p90 = self.models[0.9].predict(X_imputed)
        
        return {
            "P10": p10,
            "P50": p50,
            "P90": p90
        }

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating LightGBM Quantile Regressors...")
        preds_dict = self.predict(X)
        p10 = preds_dict["P10"]
        p50 = preds_dict["P50"]
        p90 = preds_dict["P90"]
        
        quantile_preds = {0.1: p10, 0.9: p90}
        metrics = evaluate_all_metrics(y, p50, quantile_preds)
        
        metrics["Pinball_Loss_0.5"] = evaluate_all_metrics(y, p50, {0.5: p50})["Pinball_Loss_0.5"]
        
        metrics["Pinball_Loss_0.10"] = metrics.pop("Pinball_Loss_0.1")
        metrics["Pinball_Loss_0.50"] = metrics.pop("Pinball_Loss_0.5")
        metrics["Pinball_Loss_0.90"] = metrics.pop("Pinball_Loss_0.9")
        
        metrics["Pinball_Loss_Average"] = (metrics["Pinball_Loss_0.10"] + metrics["Pinball_Loss_0.50"] + metrics["Pinball_Loss_0.90"]) / 3.0
        
        return metrics

    def save(self, filepath: str) -> None:
        logger.info(f"Saving LightGBM Quantile Regressors to {filepath}...")
        parent = os.path.dirname(filepath) if filepath else ""
        
        joblib.dump(self.models[0.1], os.path.join(parent, "lightgbm_q10.joblib"))
        joblib.dump(self.models[0.5], os.path.join(parent, "lightgbm_q50.joblib"))
        joblib.dump(self.models[0.9], os.path.join(parent, "lightgbm_q90.joblib"))
        
        artifacts_dir = "artifacts/models"
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(self.models[0.1], os.path.join(artifacts_dir, "lightgbm_q10.joblib"))
        joblib.dump(self.models[0.5], os.path.join(artifacts_dir, "lightgbm_q50.joblib"))
        joblib.dump(self.models[0.9], os.path.join(artifacts_dir, "lightgbm_q90.joblib"))
        
        state = {
            "feature_cols": self.feature_cols,
            "impute_values": self.impute_values,
            "is_fitted": self.is_fitted,
            "hyperparams": self.get_hyperparameters()
        }
        joblib.dump(state, filepath)
        logger.info("LightGBM Quantile Regressors successfully saved.")

    def load(self, filepath: str) -> None:
        logger.info(f"Loading LightGBM Quantile Regressors from {filepath}...")
        parent = os.path.dirname(filepath) if filepath else ""
        
        state = joblib.load(filepath)
        self.feature_cols = state["feature_cols"]
        self.impute_values = state.get("impute_values")
        self.is_fitted = state["is_fitted"]
        hparams = state["hyperparams"]
        self.learning_rate = hparams.get("learning_rate", 0.05)
        self.n_estimators = hparams.get("n_estimators", 100)
        self.num_leaves = hparams.get("num_leaves", 31)
        self.random_state = hparams.get("random_state", 42)
        
        self.models[0.1] = joblib.load(os.path.join(parent, "lightgbm_q10.joblib"))
        self.models[0.5] = joblib.load(os.path.join(parent, "lightgbm_q50.joblib"))
        self.models[0.9] = joblib.load(os.path.join(parent, "lightgbm_q90.joblib"))
        logger.info("LightGBM Quantile Regressors successfully loaded.")

    def get_model_name(self) -> str:
        return "lightgbm_quantile"

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "random_state": self.random_state,
            **self.kwargs
        }
