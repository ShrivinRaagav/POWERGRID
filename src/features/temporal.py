import pandas as pd
import numpy as np
from src.config.settings import DATE_COL
from src.utils.helpers import setup_logger

logger = setup_logger("temporal")

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based and cyclical features from the Date column.
    
    Features created:
    - Year (int)
    - Month (int)
    - WeekOfYear (int)
    - DayOfWeek (int)
    - Is_Quarter_End (int, 0/1)
    - Month_Sin, Month_Cos (cyclical month)
    - Week_Sin, Week_Cos (cyclical week)
    """
    logger.info("Extracting temporal features from Date column...")
    df_out = df.copy()
    
    if DATE_COL not in df_out.columns:
        logger.warning(f"Date column '{DATE_COL}' not found in dataframe. Skipping temporal feature extraction.")
        return df_out
        
    # Ensure Date column is datetime
    dates = pd.to_datetime(df_out[DATE_COL])
    
    # Extract standard integer features
    df_out["Year"] = dates.dt.year
    df_out["Month"] = dates.dt.month
    df_out["WeekOfYear"] = dates.dt.isocalendar().week.astype(np.int32)
    df_out["DayOfWeek"] = dates.dt.dayofweek
    df_out["Is_Quarter_End"] = dates.dt.is_quarter_end.astype(np.int32)
    
    # Extract Cyclical Features (Sine and Cosine transformations)
    # This helps models recognize that Month 12 (December) is adjacent to Month 1 (January)
    df_out["Month_Sin"] = np.sin(2 * np.pi * df_out["Month"] / 12.0)
    df_out["Month_Cos"] = np.cos(2 * np.pi * df_out["Month"] / 12.0)
    
    df_out["Week_Sin"] = np.sin(2 * np.pi * df_out["WeekOfYear"] / 52.0)
    df_out["Week_Cos"] = np.cos(2 * np.pi * df_out["WeekOfYear"] / 52.0)
    
    logger.info("Temporal and cyclical features successfully extracted.")
    return df_out
