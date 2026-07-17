import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils.helpers import setup_logger

logger = setup_logger("variance_filter")

def calculate_variances(df: pd.DataFrame, exclude_cols: List[str]) -> pd.Series:
    """Computes variances for numerical columns, excluding non-feature columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in num_cols if c not in exclude_cols]
    
    if not feature_cols:
        return pd.Series(dtype=np.float64)
        
    return df[feature_cols].var()

def find_low_variance_features(
    df: pd.DataFrame,
    threshold: float = 0.01,
    exclude_cols: List[str] = None
) -> Tuple[List[str], pd.Series]:
    """
    Identifies constant and near-zero variance features.
    
    Parameters:
    df (pd.DataFrame): Input dataset.
    threshold (float): Variance threshold below which features are flagged.
    exclude_cols (list): Columns that must not be removed.
    
    Returns:
    Tuple:
      - List of low-variance feature names.
      - Series containing all feature variances.
    """
    exclude_cols = exclude_cols or []
    variances = calculate_variances(df, exclude_cols)
    
    if variances.empty:
        return [], variances
        
    # Constant features have variance close to 0
    constant_features = list(variances[variances == 0.0].index)
    low_var_features = list(variances[(variances > 0.0) & (variances < threshold)].index)
    
    removed = constant_features + low_var_features
    logger.info(f"Variance filter: Identified {len(constant_features)} constant and {len(low_var_features)} near-zero variance features.")
    
    return removed, variances
