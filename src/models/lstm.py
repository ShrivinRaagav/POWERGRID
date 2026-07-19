import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from src.models.base_model import BaseForecastModel
from src.models.registry import register_model
from src.evaluation.metrics import evaluate_all_metrics
from src.utils.helpers import setup_logger

logger = setup_logger("lstm")

@register_model("lstm")
class LSTMForecastModel(BaseForecastModel):
    """
    LSTM forecasting model using TensorFlow/Keras.
    Prepares sequence windows automatically from the dataset.
    Architecture: Input -> LSTM -> Dropout -> Dense(1)
    """
    def __init__(self, window_size: int = 4, lstm_units: int = 64, dropout: float = 0.2, epochs: int = 30, batch_size: int = 16, learning_rate: float = 0.001, **kwargs):
        self.window_size = int(window_size)
        self.lstm_units = int(lstm_units)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.kwargs = kwargs
        
        self.model = None
        self.feature_cols = None
        self.impute_values = None
        self.last_history = None
        self.is_fitted = False

    def _to_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None, prepend_history: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepares sequences for LSTM. Prepends history to keep correct sequence count if provided."""
        if prepend_history is not None:
            X_concat = pd.concat([prepend_history.iloc[-self.window_size:], X], axis=0)
            X_arr = X_concat.values
        else:
            X_arr = X.values

        X_seq = []
        for i in range(len(X_arr) - self.window_size):
            X_seq.append(X_arr[i : i + self.window_size])
        
        y_seq = None
        if y is not None:
            y_arr = y.values
            y_seq = y_arr[self.window_size:]
                
        return np.array(X_seq), (np.array(y_seq) if y_seq is not None else None)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> None:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.optimizers import Adam
        
        logger.info("Preparing LSTM sequence datasets...")
        self.feature_cols = list(X_train.columns)
        num_features = len(self.feature_cols)
        
        # Calculate training medians for robust imputation
        self.impute_values = X_train.median().fillna(0.0)
        X_train_imputed = X_train.fillna(self.impute_values)
        
        # Save last window_size samples from X_train_imputed to use as history for evaluation/prediction
        self.last_history = X_train_imputed.iloc[-self.window_size:]
        
        X_train_seq, y_train_seq = self._to_sequences(X_train_imputed, y_train)
        
        # Validation data sequences
        if X_val is not None and y_val is not None:
            X_val_imputed = X_val.fillna(self.impute_values)
            X_val_seq, y_val_seq = self._to_sequences(X_val_imputed, y_val)
            validation_data = (X_val_seq, y_val_seq)
            # Update history to be the end of validation set
            self.last_history = X_val_imputed.iloc[-self.window_size:]
        else:
            validation_data = None
            
        logger.info("Building LSTM model architecture...")
        self.model = Sequential([
            Input(shape=(self.window_size, num_features)),
            LSTM(units=self.lstm_units),
            Dropout(rate=self.dropout),
            Dense(units=1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse"
        )
        
        logger.info(f"Fitting LSTM model for {self.epochs} epochs (batch size: {self.batch_size})...")
        self.model.fit(
            X_train_seq, y_train_seq,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("LSTM training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("LSTM model is not fitted yet.")
        
        # Ensure column order is aligned and imputed
        X_aligned = X[self.feature_cols]
        X_imputed = X_aligned.fillna(self.impute_values)
        
        # Prepend last history to X to generate exactly len(X) forecasts
        if self.last_history is not None and len(self.last_history) == self.window_size:
            X_concat = pd.concat([self.last_history, X_imputed], axis=0)
        else:
            padding = pd.DataFrame([X_imputed.iloc[0]] * self.window_size, columns=X_imputed.columns)
            X_concat = pd.concat([padding, X_imputed], axis=0)
            
        X_seq, _ = self._to_sequences(X_concat)
        
        # Generate predictions
        preds = self.model.predict(X_seq, verbose=0).flatten()
        return preds

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info("Evaluating LSTM...")
        preds = self.predict(X)
        return evaluate_all_metrics(y, preds)

    def save(self, filepath: str) -> None:
        logger.info(f"Saving LSTM state to {filepath}...")
        
        keras_path = filepath + ".keras"
        if filepath.endswith(".keras") or filepath.endswith(".h5"):
            keras_path = filepath
            
        self.model.save(keras_path)
        
        # Save metadata and placeholder joblib
        meta_path = filepath + ".meta"
        metadata = {
            "window_size": self.window_size,
            "lstm_units": self.lstm_units,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "is_fitted": self.is_fitted,
            "impute_values": self.impute_values,
            "last_history": self.last_history,
            "feature_cols": self.feature_cols,
            "hyperparams": self.get_hyperparameters()
        }
        joblib.dump(metadata, meta_path)
        
        # If we save the checkpoint (which doesn't end with .keras), write dummy file at filepath
        if not (filepath.endswith(".keras") or filepath.endswith(".h5")):
            joblib.dump({"is_lstm": True, "keras_path": keras_path}, filepath)
            
        # Save copy to artifacts/models/lstm.keras
        artifacts_path = "artifacts/models/lstm.keras"
        os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
        self.model.save(artifacts_path)
        joblib.dump(metadata, artifacts_path + ".meta")
        
        logger.info("LSTM successfully saved.")

    def load(self, filepath: str) -> None:
        logger.info(f"Loading LSTM state from {filepath}...")
        from tensorflow.keras.models import load_model
        
        meta_path = filepath + ".meta"
        metadata = joblib.load(meta_path)
        
        self.window_size = metadata["window_size"]
        self.lstm_units = metadata["lstm_units"]
        self.dropout = metadata["dropout"]
        self.epochs = metadata["epochs"]
        self.batch_size = metadata["batch_size"]
        self.learning_rate = metadata["learning_rate"]
        self.is_fitted = metadata["is_fitted"]
        self.impute_values = metadata.get("impute_values")
        self.last_history = metadata["last_history"]
        self.feature_cols = metadata["feature_cols"]
        
        keras_path = filepath + ".keras"
        if filepath.endswith(".keras") or filepath.endswith(".h5"):
            keras_path = filepath
            
        self.model = load_model(keras_path)
        logger.info("LSTM successfully loaded.")

    def get_model_name(self) -> str:
        return "lstm"

    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "lstm_units": self.lstm_units,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            **self.kwargs
        }
