import pandas as pd
import numpy as np
from typing import List, Union, Optional
from src.utils.helpers import setup_logger

logger = setup_logger("time_series_utils")

def prepare_time_series(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    date_col: str = "Date",
    target_col: str = "Quantity_Required",
    freq: str = "W-SUN",
    fill_method: str = "zero"
) -> pd.DataFrame:
    """
    Converts a processed POWERGRID dataset into time-series format.
    - Chronologically sorts the data.
    - Resolves missing weekly timestamps by reindexing.
    - Imputes missing target values based on the configurable fill_method.
    - Preserves categorical group structures.

    Parameters:
    df (pd.DataFrame): Input dataframe.
    group_cols (list, optional): List of columns to group time series by.
    date_col (str): Date column name.
    target_col (str): Target column name (Quantity_Required).
    freq (str): Date frequency (e.g. 'W-SUN' for weekly).
    fill_method (str): Method to fill missing timestamps ('zero', 'ffill', 'interpolate').

    Returns:
    pd.DataFrame: Formatted time-series dataset.
    """
    if df.empty:
        logger.warning("Empty dataframe provided to prepare_time_series.")
        return df.copy()

    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col])
    
    # Define date range bounds based on the dataset
    min_date = df_out[date_col].min()
    max_date = df_out[date_col].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq=freq)
    
    logger.info(f"Preparing time series from {min_date.date()} to {max_date.date()} (Freq: {freq}, Fill Method: {fill_method})")
    
    if not group_cols:
        # Global aggregation
        logger.info("No grouping columns specified. Aggregating globally...")
        # Sum targets if there are multiple entries per date
        df_agg = df_out.groupby(date_col)[target_col].sum().reset_index()
        df_agg = df_agg.sort_values(by=date_col).set_index(date_col)
        
        # Reindex to fill missing timestamps
        df_agg = df_agg.reindex(all_dates)
        
        # Apply fill method
        if fill_method == "zero":
            df_agg[target_col] = df_agg[target_col].fillna(0.0)
        elif fill_method == "ffill":
            df_agg[target_col] = df_agg[target_col].ffill().fillna(0.0)
        elif fill_method == "interpolate":
            df_agg[target_col] = df_agg[target_col].interpolate(method="linear").fillna(0.0)
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")
            
        df_agg = df_agg.reset_index().rename(columns={"index": date_col})
        return df_agg

    # Grouped time series
    logger.info(f"Grouping time-series by columns: {group_cols}")
    grouped = df_out.groupby(group_cols)
    results = []
    
    for name, group in grouped:
        # Sum target values in case of duplicate dates within same group
        group_agg = group.groupby(date_col)[target_col].sum().reset_index()
        group_agg = group_agg.sort_values(by=date_col).set_index(date_col)
        
        # Reindex to full date range
        group_agg = group_agg.reindex(all_dates)
        
        # Apply fill method
        if fill_method == "zero":
            group_agg[target_col] = group_agg[target_col].fillna(0.0)
        elif fill_method == "ffill":
            group_agg[target_col] = group_agg[target_col].ffill().fillna(0.0)
        elif fill_method == "interpolate":
            group_agg[target_col] = group_agg[target_col].interpolate(method="linear").fillna(0.0)
        else:
            raise ValueError(f"Unknown fill_method: {fill_method}")
            
        group_agg = group_agg.reset_index().rename(columns={"index": date_col})
        
        # Restore grouping column values
        if len(group_cols) == 1:
            group_agg[group_cols[0]] = name
        else:
            for idx, col in enumerate(group_cols):
                group_agg[col] = name[idx]
                
        results.append(group_agg)
        
    final_df = pd.concat(results, ignore_index=True)
    # Sort chronologically by date and then by group columns
    final_df = final_df.sort_values(by=[date_col] + group_cols).reset_index(drop=True)
    logger.info(f"Time-series preparation complete. Row count: {len(final_df)}")
    return final_df
