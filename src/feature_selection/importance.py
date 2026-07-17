import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from src.utils.helpers import setup_logger

logger = setup_logger("importance_calculator")

def calculate_feature_importances(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: List[str],
    method: str = "mutual_info",
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Computes feature importance scores using Mutual Information or a simple Random Forest.
    
    Parameters:
    df (pd.DataFrame): Dataset.
    target_col (str): Target column name (Quantity_Required).
    exclude_cols (list): Non-feature columns.
    method (str): "mutual_info" or "random_forest".
    random_seed (int): Random seed for reproducibility.
    
    Returns:
    Dict[str, float]: Mapping from feature names to importance scores.
    """
    # Exclude categorical columns or columns that shouldn't be trained on
    num_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in num_cols if c not in exclude_cols and c != target_col]
    
    if not feature_cols:
        logger.warning("No feature columns available to compute importances.")
        return {}
        
    # Drop rows with NaN targets or NaN features
    df_clean = df[[target_col] + feature_cols].dropna()
    
    if df_clean.empty:
        logger.warning("Empty dataset after dropping NaNs. Setting importances to 0.0.")
        return {col: 0.0 for col in feature_cols}
        
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    importances = {}
    logger.info(f"Computing feature importances using method: '{method}'...")
    
    if method == "mutual_info":
        try:
            scores = mutual_info_regression(X, y, random_state=random_seed)
            importances = {col: float(score) for col, score in zip(feature_cols, scores)}
        except Exception as e:
            logger.error(f"Error computing Mutual Information: {e}. Falling back to uniform importances.")
            importances = {col: 1.0 for col in feature_cols}
            
    elif method == "random_forest":
        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=random_seed, n_jobs=-1)
            rf.fit(X, y)
            importances = {col: float(imp) for col, imp in zip(feature_cols, rf.feature_importances_)}
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}. Falling back to Mutual Information.")
            # Recursive fallback to mutual_info
            return calculate_feature_importances(df, target_col, exclude_cols, method="mutual_info", random_seed=random_seed)
            
    else:
        raise ValueError(f"Unknown importance method: {method}")
        
    # Normalize scores to sum to 1 (or keep raw scores for ranking)
    total = sum(importances.values())
    if total > 0.0:
        importances_norm = {k: v / total for k, v in importances.items()}
    else:
        importances_norm = importances
        
    logger.info(f"Feature importance calculation complete.")
    return importances_norm
