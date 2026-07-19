import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.neural_network import MLPRegressor
from src.models.base_model import BaseForecastModel
from src.models.registry import register_model
from src.evaluation.metrics import evaluate_all_metrics
from src.utils.helpers import setup_logger

logger = setup_logger("mlp")

@register_model("mlp")
class MLPForecastModel(BaseForecastModel):
    """
    Multi-Layer Perceptron (MLP) forecasting model for material demand prediction.
    Obtains all network and learning configurations from config.yaml.
    """
    def __init__(self, hidden_layer_sizes: List[int] = [100, 50], activation: str = "relu", learning_rate: str = "constant", learning_rate_init: float = 0.001, max_iter: int = 500, batch_size: str = "auto", random_state: int = 42, **kwargs):
        self.hidden_layer_sizes = list(hidden_layer_sizes)
        self.activation = activation
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.kwargs = kwargs
        
        parsed_batch_size = self.batch_size
        if isinstance(self.batch_size, str) and self.batch_size.isdigit():
            parsed_batch_size = int(self.batch_size)
            
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            batch_size=parsed_batch_size,
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
        logger.info("Training MLPRegressor...")
        self.feature_cols = list(X_train.columns)
        
        # Calculate training medians for robust imputation
        self.impute_values = X_train.median().fillna(0.0)
        X_train_imputed = X_train.fillna(self.impute_values)
        
        self.model.fit(X_train_imputed, y_train)
        self.is_fitted = True
        logger.info("MLPRegressor training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MLPRegressor is not fitted yet.")
        X_aligned = X[self.feature_cols]
        X_imputed = X_aligned.fillna(self.impute_values)
        return self.model.predict(X_imputed)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating MLPRegressor...")
        preds = self.predict(X)
        return evaluate_all_metrics(y, preds)

    def save(self, filepath: str) -> None:
        logger.info(f"Saving MLPRegressor to {filepath}...")
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
        artifacts_path = "artifacts/models/mlp.joblib"
        os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
        joblib.dump(state, artifacts_path)
        logger.info("MLPRegressor successfully saved.")

    def load(self, filepath: str) -> None:
        logger.info(f"Loading MLPRegressor from {filepath}...")
        state = joblib.load(filepath)
        self.model = state["model"]
        self.feature_cols = state["feature_cols"]
        self.impute_values = state.get("impute_values")
        self.is_fitted = state["is_fitted"]
        # Sync attributes with loaded hyperparameters
        hparams = state["hyperparams"]
        self.hidden_layer_sizes = list(hparams.get("hidden_layer_sizes", [100, 50]))
        self.activation = hparams.get("activation", "relu")
        self.learning_rate = hparams.get("learning_rate", "constant")
        self.learning_rate_init = hparams.get("learning_rate_init", 0.001)
        self.max_iter = hparams.get("max_iter", 500)
        self.batch_size = hparams.get("batch_size", "auto")
        self.random_state = hparams.get("random_state", 42)

    def get_model_name(self) -> str:
        return "mlp"

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            **self.kwargs
        }
