import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.config.settings import (
    DATE_COL, TARGET_COL, CATEGORICAL_COLS, NUMERICAL_COLS,
    VALID_REGIONS, VALID_PROJECT_PHASES, VALID_MATERIAL_TYPES
)
from src.utils.helpers import setup_logger

logger = setup_logger("cleaner")

class DataCleaner:
    """
    Modular cleaner class that:
    1. Converts dates and handles invalid date rows.
    2. Removes exact duplicates.
    3. Handles missing values via median (numeric) and mode (categorical) imputation.
    4. Handles negative inventory and demand by setting them to zero.
    5. Detects and caps outliers using IQR thresholds.
    
    Prevents data leakage by fitting parameters on the training set and transforming validation/test sets.
    """
    def __init__(self):
        self.medians: Dict[str, float] = {}
        self.modes: Dict[str, str] = {}
        self.outlier_bounds: Dict[str, Tuple[float, float]] = {}
        self.is_fit = False

    def fit(self, df: pd.DataFrame):
        """
        Fits cleaning parameters on the training dataset.
        """
        logger.info("Fitting cleaner on training dataset...")
        
        # 1. Compute Medians for numeric columns
        for col in NUMERICAL_COLS:
            # Filter out negative values to get a clean baseline median
            valid_vals = df[col][df[col] >= 0] if col in [TARGET_COL, "Historical_Demand", "Current_Inventory"] else df[col]
            median_val = valid_vals.median()
            self.medians[col] = median_val if not pd.isna(median_val) else 0.0
            
            # Compute IQR outlier bounds
            q1 = valid_vals.quantile(0.25)
            q3 = valid_vals.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.outlier_bounds[col] = (lower_bound, upper_bound)
            
        if TARGET_COL in df.columns:
            valid_vals = df[TARGET_COL][df[TARGET_COL] >= 0]
            self.medians[TARGET_COL] = valid_vals.median() if not pd.isna(valid_vals.median()) else 0.0
            q1 = valid_vals.quantile(0.25)
            q3 = valid_vals.quantile(0.75)
            iqr = q3 - q1
            self.outlier_bounds[TARGET_COL] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            
        # 2. Compute Modes for categorical columns
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                mode_series = df[col].dropna().mode()
                self.modes[col] = mode_series[0] if not mode_series.empty else "UNKNOWN"
                
        self.is_fit = True
        logger.info("Cleaner fitted successfully.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and normalizes the dataframe using the fitted parameters.
        """
        if not self.is_fit:
            raise ValueError("Cleaner must be fit on training data before transforming.")
            
        logger.info(f"Transforming dataset with shape {df.shape}...")
        cleaned_df = df.copy()
        
        # 1. Remove exact duplicates
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        dup_removed = initial_rows - len(cleaned_df)
        if dup_removed > 0:
            logger.info(f"Removed {dup_removed} exact duplicate records.")
            
        # 2. Validate and convert Date formats, drop rows with unparseable dates
        if DATE_COL in cleaned_df.columns:
            initial_len = len(cleaned_df)
            cleaned_df[DATE_COL] = pd.to_datetime(cleaned_df[DATE_COL], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=[DATE_COL])
            invalid_dates = initial_len - len(cleaned_df)
            if invalid_dates > 0:
                logger.warning(f"Dropped {invalid_dates} rows with unparseable or null dates.")
                
        # 3. Clean Categorical columns: handle missing values and invalid types
        for col in CATEGORICAL_COLS:
            if col in cleaned_df.columns:
                # Force string type
                cleaned_df[col] = cleaned_df[col].astype(str)
                # Fill missing/NaN (which might now be 'nan' string)
                cleaned_df[col] = cleaned_df[col].replace({'nan': np.nan, 'None': np.nan})
                mode_val = self.modes.get(col, "UNKNOWN")
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                
                # Check for out-of-bounds/invalid category names
                if col == "Region":
                    invalid_mask = ~cleaned_df[col].isin(VALID_REGIONS)
                    if invalid_mask.any():
                        logger.warning(f"Replacing {invalid_mask.sum()} invalid region values with mode: '{mode_val}'.")
                        cleaned_df.loc[invalid_mask, col] = mode_val
                elif col == "Project_Phase":
                    invalid_mask = ~cleaned_df[col].isin(VALID_PROJECT_PHASES)
                    if invalid_mask.any():
                        logger.warning(f"Replacing {invalid_mask.sum()} invalid project phase values with mode: '{mode_val}'.")
                        cleaned_df.loc[invalid_mask, col] = mode_val
                elif col == "Material_Type":
                    invalid_mask = ~cleaned_df[col].isin(VALID_MATERIAL_TYPES)
                    if invalid_mask.any():
                        logger.warning(f"Replacing {invalid_mask.sum()} invalid material type values with mode: '{mode_val}'.")
                        cleaned_df.loc[invalid_mask, col] = mode_val
                
        # 4. Clean Numerical columns: handle missing values, negatives, and outliers
        for col in NUMERICAL_COLS:
            if col in cleaned_df.columns:
                # Convert to numeric
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                # First handle negative values
                if col in ["Current_Inventory", "Historical_Demand"]:
                    neg_mask = cleaned_df[col] < 0
                    if neg_mask.any():
                        logger.warning(f"Resetting {neg_mask.sum()} negative values in {col} to 0.")
                        cleaned_df.loc[neg_mask, col] = 0.0
                        
                # Fill missing values
                nan_mask = cleaned_df[col].isna()
                if nan_mask.any():
                    logger.info(f"Imputing {nan_mask.sum()} missing values in {col} with median ({self.medians[col]:.2f}).")
                    cleaned_df[col] = cleaned_df[col].fillna(self.medians[col])
                    
                # Outlier capping (Winsorization)
                if col in self.outlier_bounds:
                    lower, upper = self.outlier_bounds[col]
                    # Ensure positive values check for inventory/demand
                    if col in ["Current_Inventory", "Historical_Demand", "Lead_Time_Days"]:
                        lower = max(0.0, lower)
                    
                    outlier_mask = (cleaned_df[col] < lower) | (cleaned_df[col] > upper)
                    if outlier_mask.any():
                        logger.info(f"Capping {outlier_mask.sum()} outliers in {col} to range [{lower:.2f}, {upper:.2f}].")
                        cleaned_df[col] = cleaned_df[col].clip(lower=lower, upper=upper)
                        
        # 5. Clean Target column
        if TARGET_COL in cleaned_df.columns:
            cleaned_df[TARGET_COL] = pd.to_numeric(cleaned_df[TARGET_COL], errors='coerce')
            
            # Negatives check
            neg_mask = cleaned_df[TARGET_COL] < 0
            if neg_mask.any():
                logger.warning(f"Resetting {neg_mask.sum()} negative values in {TARGET_COL} to 0.")
                cleaned_df.loc[neg_mask, TARGET_COL] = 0.0
                
            # Missing check
            nan_mask = cleaned_df[TARGET_COL].isna()
            if nan_mask.any():
                logger.info(f"Imputing {nan_mask.sum()} missing values in {TARGET_COL} with median ({self.medians[TARGET_COL]:.2f}).")
                cleaned_df[TARGET_COL] = cleaned_df[TARGET_COL].fillna(self.medians[TARGET_COL])
                
            # Outlier capping
            if TARGET_COL in self.outlier_bounds:
                lower, upper = self.outlier_bounds[TARGET_COL]
                lower = max(0.0, lower)
                outlier_mask = (cleaned_df[TARGET_COL] < lower) | (cleaned_df[TARGET_COL] > upper)
                if outlier_mask.any():
                    logger.info(f"Capping {outlier_mask.sum()} outliers in {TARGET_COL} to range [{lower:.2f}, {upper:.2f}].")
                    cleaned_df[TARGET_COL] = cleaned_df[TARGET_COL].clip(lower=lower, upper=upper)

        logger.info(f"Transformation complete. Output shape: {cleaned_df.shape}")
        return cleaned_df
