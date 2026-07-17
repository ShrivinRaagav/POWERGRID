import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils.helpers import setup_logger

logger = setup_logger("feature_selection_utils")

def find_duplicated_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Identifies columns that have identical values.
    
    Parameters:
    df (pd.DataFrame): Input dataset.
    exclude_cols (list): Protected columns.
    
    Returns:
    List[str]: List of duplicated feature names (all but the first occurrence).
    """
    exclude_cols = exclude_cols or []
    
    # Select only feature columns
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if len(feature_cols) < 2:
        return []
        
    duplicated_cols = []
    
    # Loop over features and check duplicates
    # A quick way to find duplicate columns is to check column values equality
    for i in range(len(feature_cols)):
        col1 = feature_cols[i]
        if col1 in duplicated_cols:
            continue
            
        for j in range(i + 1, len(feature_cols)):
            col2 = feature_cols[j]
            if col2 in duplicated_cols:
                continue
                
            # If dtype is object or categorical, convert to string for comparison
            # or compare directly
            try:
                if df[col1].equals(df[col2]):
                    duplicated_cols.append(col2)
                    logger.info(f"Duplicate check: '{col2}' is identical to '{col1}' and will be removed.")
            except Exception as e:
                # Handle potential comparisons error on mixed dtypes
                logger.debug(f"Error comparing '{col1}' and '{col2}': {e}")
                
    return duplicated_cols
