import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils.helpers import setup_logger

logger = setup_logger("feature_selection_utils")

def find_duplicated_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Identifies columns that have 100% identical values.
    
    Parameters:
    df (pd.DataFrame): Input dataset.
    exclude_cols (list): Protected columns.
    
    Returns:
    List[str]: List of duplicated feature names (all but the first occurrence).
    """
    exclude_cols = exclude_cols or []
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if len(feature_cols) < 2:
        return []
        
    df_feats = df[feature_cols].copy()
    dup_mask = df_feats.T.duplicated(keep="first")
    candidate_dups = list(df_feats.columns[dup_mask])
    
    # Protect distinct signal decomposition and time-series feature names from false duplicate drops
    protected_prefixes = ("DWT_", "EMD_", "Classical_", "Lag_", "Rolling_", "Seasonal_", "Trend_", "Signal_")
    duplicated_cols = [c for c in candidate_dups if not any(c.startswith(p) for p in protected_prefixes)]
    
    for col in duplicated_cols:
        logger.info(f"Duplicate check: '{col}' is identical to a preceding feature and will be removed.")
        
    return duplicated_cols
