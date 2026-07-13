import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.config.settings import MODELS_DIR, NUMERICAL_COLS
from src.utils.helpers import setup_logger

logger = setup_logger("scaler")

class NumericalScaler:
    """
    StandardScaler wrapper that fits scaling parameters on Training data
    and applies them to Validation and Test data, serializing the state to disk.
    """
    def __init__(self, save_dir: Path = MODELS_DIR):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.numerical_cols = NUMERICAL_COLS
        self.is_fit = False

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler on the numerical columns of the training dataset.
        """
        logger.info(f"Fitting StandardScaler on {len(df)} rows...")
        cols_to_scale = [c for c in self.numerical_cols if c in df.columns]
        if not cols_to_scale:
            logger.warning("No numerical columns found to scale.")
            self.is_fit = True
            return
            
        self.numerical_cols = cols_to_scale
        
        # Fit scaler
        self.scaler.fit(df[self.numerical_cols])
        self.is_fit = True
        logger.info("Scaler successfully fitted.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies standard scaling to the numerical columns.
        """
        if not self.is_fit:
            raise ValueError("Scaler must be fit on training data before transforming.")
            
        if not self.numerical_cols:
            return df.copy()
            
        logger.info("Scaling numerical features...")
        df_out = df.copy()
        
        scaled_array = self.scaler.transform(df_out[self.numerical_cols])
        df_out[self.numerical_cols] = scaled_array
        
        return df_out

    def save(self, filename: str = "numerical_scaler.joblib"):
        """Serializes the scaler state to disk."""
        filepath = self.save_dir / filename
        state = {
            "numerical_cols": self.numerical_cols,
            "scaler": self.scaler,
            "is_fit": self.is_fit
        }
        joblib.dump(state, filepath)
        logger.info(f"Scaler saved to {filepath}")

    def load(self, filename: str = "numerical_scaler.joblib"):
        """Loads serialized scaler state from disk."""
        filepath = self.save_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"No scaler file found at {filepath}")
        state = joblib.load(filepath)
        self.numerical_cols = state["numerical_cols"]
        self.scaler = state["scaler"]
        self.is_fit = state["is_fit"]
        logger.info(f"Scaler loaded from {filepath}")
