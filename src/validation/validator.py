import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from src.config.settings import (
    VALIDATION_REPORT_PATH, DATE_COL, TARGET_COL, CATEGORICAL_COLS, NUMERICAL_COLS,
    VALID_REGIONS, VALID_PROJECT_PHASES
)
from src.utils.helpers import setup_logger

logger = setup_logger("validator")

def run_data_validation(df: pd.DataFrame, stage: str = "raw") -> pd.DataFrame:
    """
    Validates a dataset against predefined rules.
    Generates a structured report detailing pass/fail status and violation counts.
    
    Parameters:
    df (pd.DataFrame): Dataset to validate
    stage (str): The processing stage (e.g., 'raw', 'cleaned') for report metadata
    
    Returns:
    pd.DataFrame: A validation report summary.
    """
    logger.info(f"Running data validation checks for stage: {stage} on {len(df)} rows...")
    
    report_rows = []
    
    # 1. Missing Values Check
    missing_counts = df.isnull().sum()
    for col in df.columns:
        cnt = missing_counts[col]
        status = "PASS" if cnt == 0 else "FAIL"
        report_rows.append({
            "Stage": stage,
            "Check_Name": "Missing Values Check",
            "Column": col,
            "Status": status,
            "Violation_Count": int(cnt),
            "Description": f"Checks if column {col} contains null values."
        })
        
    # 2. Duplicate Records Check
    dup_count = df.duplicated().sum()
    report_rows.append({
        "Stage": stage,
        "Check_Name": "Duplicate Records Check",
        "Column": "ALL_COLUMNS",
        "Status": "PASS" if dup_count == 0 else "FAIL",
        "Violation_Count": int(dup_count),
        "Description": "Checks for identical rows in the dataset."
    })
    
    # 3. Negative Inventory Check
    if "Current_Inventory" in df.columns:
        # handle potential NaNs before comparisons
        neg_inv_count = (df["Current_Inventory"] < 0).sum()
        report_rows.append({
            "Stage": stage,
            "Check_Name": "Negative Inventory Check",
            "Column": "Current_Inventory",
            "Status": "PASS" if neg_inv_count == 0 else "FAIL",
            "Violation_Count": int(neg_inv_count),
            "Description": "Checks for negative values in the current inventory column."
        })
        
    # 4. Negative Demand Check
    if "Quantity_Required" in df.columns:
        neg_dem_count = (df["Quantity_Required"] < 0).sum()
        report_rows.append({
            "Stage": stage,
            "Check_Name": "Negative Demand Check",
            "Column": "Quantity_Required",
            "Status": "PASS" if neg_dem_count == 0 else "FAIL",
            "Violation_Count": int(neg_dem_count),
            "Description": "Checks for negative values in the quantity required (demand) column."
        })
        
    if "Historical_Demand" in df.columns:
        neg_hist_count = (df["Historical_Demand"] < 0).sum()
        report_rows.append({
            "Stage": stage,
            "Check_Name": "Negative Historical Demand Check",
            "Column": "Historical_Demand",
            "Status": "PASS" if neg_hist_count == 0 else "FAIL",
            "Violation_Count": int(neg_hist_count),
            "Description": "Checks for negative values in the historical demand column."
        })
        
    # 5. Invalid Dates Check
    invalid_dates = 0
    if DATE_COL in df.columns:
        # Convert date to datetime; errors='coerce' turns invalid dates to NaT
        temp_dates = pd.to_datetime(df[DATE_COL], errors='coerce')
        invalid_dates = temp_dates.isna().sum() - df[DATE_COL].isna().sum()
        # Also check if dates fall within a reasonable range: 2020-01-01 to 2030-12-31
        valid_date_mask = temp_dates.notna()
        out_of_bounds = ((temp_dates[valid_date_mask] < pd.Timestamp("2020-01-01")) | 
                         (temp_dates[valid_date_mask] > pd.Timestamp("2030-12-31"))).sum()
        invalid_dates += out_of_bounds
        
        report_rows.append({
            "Stage": stage,
            "Check_Name": "Invalid Date Formats or Range",
            "Column": DATE_COL,
            "Status": "PASS" if invalid_dates == 0 else "FAIL",
            "Violation_Count": int(invalid_dates),
            "Description": "Checks if date format is parseable and falls between 2020 and 2030."
        })
        
    # 6. Invalid Regions Check
    if "Region" in df.columns:
        invalid_regs = (~df["Region"].isin(VALID_REGIONS)).sum()
        # Do not flag NaNs under invalid category (they are caught in missing check)
        invalid_regs = max(0, int(invalid_regs - df["Region"].isna().sum()))
        report_rows.append({
            "Stage": stage,
            "Check_Name": "Invalid Region Code",
            "Column": "Region",
            "Status": "PASS" if invalid_regs == 0 else "FAIL",
            "Violation_Count": invalid_regs,
            "Description": f"Checks if region matches one of the valid codes: {VALID_REGIONS}"
        })
        
    # 7. Invalid Project Phases Check
    if "Project_Phase" in df.columns:
        invalid_phases = (~df["Project_Phase"].isin(VALID_PROJECT_PHASES)).sum()
        # Adjust for NaNs
        invalid_phases = max(0, int(invalid_phases - df["Project_Phase"].isna().sum()))
        report_rows.append({
            "Stage": stage,
            "Check_Name": "Invalid Project Phase",
            "Column": "Project_Phase",
            "Status": "PASS" if invalid_phases == 0 else "FAIL",
            "Violation_Count": invalid_phases,
            "Description": f"Checks if project phase matches: {VALID_PROJECT_PHASES}"
        })
        
    report_df = pd.DataFrame(report_rows)
    logger.info(f"Validation completed. {report_df[report_df['Status'] == 'FAIL'].shape[0]} checks failed.")
    return report_df

def save_validation_report(report_df: pd.DataFrame, path: str = VALIDATION_REPORT_PATH):
    """Saves the validation report to CSV."""
    report_df.to_csv(path, index=False)
    logger.info(f"Validation report saved to: {path}")
