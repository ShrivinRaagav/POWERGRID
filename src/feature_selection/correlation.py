import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.utils.helpers import setup_logger

logger = setup_logger("correlation_filter")

def find_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.85,
    exclude_cols: Optional[List[str]] = None,
    importances: Optional[Dict[str, float]] = None,
    variances: Optional[pd.Series] = None
) -> Tuple[List[str], pd.DataFrame, List[Dict[str, Any]]]:
    """
    Identifies highly correlated features. For each correlated pair, flags the one with 
    lower importance (or lower variance as a tie-breaker) for removal.
    
    Parameters:
    df (pd.DataFrame): Dataset.
    threshold (float): Pearson correlation absolute threshold (0.0 to 1.0).
    exclude_cols (list): Columns to protect.
    importances (dict): Optional feature importance mapping for smart dropping.
    variances (pd.Series): Optional feature variances for tie-breaking.
    
    Returns:
    Tuple:
      - List of features flagged for removal.
      - Correlation matrix DataFrame.
      - List of dictionaries detailing each correlated pair.
    """
    exclude_cols = exclude_cols or []
    importances = importances or {}
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in num_cols if c not in exclude_cols]
    
    if len(feature_cols) < 2:
        return [], pd.DataFrame(), []
        
    corr_matrix = df[feature_cols].corr().abs()
    
    # Track pairs and columns to drop
    to_drop = set()
    pair_details = []
    
    # Upper triangle traversal
    upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if corr_val >= threshold:
                # Get importances and variances
                imp1 = importances.get(col1, 0.0)
                imp2 = importances.get(col2, 0.0)
                
                var1 = variances.get(col1, 0.0) if variances is not None else 0.0
                var2 = variances.get(col2, 0.0) if variances is not None else 0.0
                
                # Determine which feature to drop
                if imp1 != imp2:
                    drop_col = col2 if imp1 > imp2 else col1
                    keep_col = col1 if imp1 > imp2 else col2
                else:
                    drop_col = col2 if var1 >= var2 else col1
                    keep_col = col1 if var1 >= var2 else col2
                    
                to_drop.add(drop_col)
                pair_details.append({
                    "Feature_1": col1,
                    "Feature_2": col2,
                    "Correlation": corr_val,
                    "Keep": keep_col,
                    "Drop": drop_col,
                    "Reason": f"Corr={corr_val:.4f} (Keep {keep_col} over {drop_col} due to higher importance/variance)"
                })
                
    logger.info(f"Correlation filter: Identified {len(to_drop)} features to drop with correlation >= {threshold}.")
    return list(to_drop), corr_matrix, pair_details
import typing
# Keep typing.Any available for type hinting
Any = typing.Any
