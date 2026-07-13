import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Union
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from src.config.settings import MODELS_DIR, CATEGORICAL_COLS
from src.utils.helpers import setup_logger

logger = setup_logger("encoder")

class CategoricalEncoder:
    """
    Categorical encoder that supports:
    1. Ordinal Encoding (mapping categories to unique integers, suitable for tree-based models).
    2. One-Hot Encoding (converting categories into binary columns, suitable for SVR, MLP, LSTMs).
    
    Encoders are fit on Training data and applied to Val/Test data, and serialized via joblib.
    """
    def __init__(self, method: str = "ordinal", save_dir: Path = MODELS_DIR):
        """
        Parameters:
        method (str): 'ordinal' or 'onehot'
        save_dir (Path): directory to save fitted encoder models
        """
        self.method = method.lower()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = None
        self.is_fit = False
        self.encoded_cols: List[str] = []
        self.categorical_cols: List[str] = CATEGORICAL_COLS

    def fit(self, df: pd.DataFrame):
        """
        Fits the encoder on the training dataframe.
        """
        logger.info(f"Fitting {self.method} encoder on {len(df)} rows...")
        # Subset categorical columns present in data
        cols_to_encode = [c for c in self.categorical_cols if c in df.columns]
        if not cols_to_encode:
            logger.warning("No categorical columns found to encode.")
            self.is_fit = True
            return
            
        self.categorical_cols = cols_to_encode
        
        # Prepare inputs as string
        X_cat = df[self.categorical_cols].astype(str)
        
        if self.method == "onehot":
            self.encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=np.float32
            )
            self.encoder.fit(X_cat)
            # Create feature names
            self.encoded_cols = list(self.encoder.get_feature_names_out(self.categorical_cols))
        elif self.method == "ordinal":
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int32
            )
            self.encoder.fit(X_cat)
            self.encoded_cols = self.categorical_cols
        else:
            raise ValueError(f"Unknown encoding method: {self.method}. Choose 'ordinal' or 'onehot'.")
            
        self.is_fit = True
        logger.info(f"{self.method.capitalize()} encoder successfully fitted.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the fitted encoder to the dataframe.
        """
        if not self.is_fit:
            raise ValueError("Encoder must be fit on training data before transforming.")
            
        if not self.categorical_cols:
            return df.copy()
            
        logger.info(f"Transforming categorical features using {self.method}...")
        df_out = df.copy()
        
        # Convert columns to string, fill nulls if any (should already be cleaned)
        X_cat = df_out[self.categorical_cols].astype(str)
        
        if self.method == "onehot":
            encoded_array = self.encoder.transform(X_cat)
            encoded_df = pd.DataFrame(
                encoded_array, 
                columns=self.encoded_cols,
                index=df_out.index
            )
            # Drop original categorical columns and append encoded columns
            df_out = df_out.drop(columns=self.categorical_cols)
            df_out = pd.concat([df_out, encoded_df], axis=1)
        elif self.method == "ordinal":
            encoded_array = self.encoder.transform(X_cat)
            df_out[self.categorical_cols] = encoded_array
            
        return df_out

    def save(self, filename: str = "categorical_encoder.joblib"):
        """Serializes the encoder state to disk."""
        filepath = self.save_dir / filename
        state = {
            "method": self.method,
            "categorical_cols": self.categorical_cols,
            "encoded_cols": self.encoded_cols,
            "encoder": self.encoder,
            "is_fit": self.is_fit
        }
        joblib.dump(state, filepath)
        logger.info(f"Encoder saved to {filepath}")

    def load(self, filename: str = "categorical_encoder.joblib"):
        """Loads serialized encoder state from disk."""
        filepath = self.save_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"No encoder file found at {filepath}")
        state = joblib.load(filepath)
        self.method = state["method"]
        self.categorical_cols = state["categorical_cols"]
        self.encoded_cols = state["encoded_cols"]
        self.encoder = state["encoder"]
        self.is_fit = state["is_fit"]
        logger.info(f"Encoder loaded from {filepath}")
