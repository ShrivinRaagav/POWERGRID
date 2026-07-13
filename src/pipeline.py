import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.config.settings import (
    RAW_DATA_PATH, PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, PROCESSED_TEST_PATH,
    PROCESSED_DATA_DIR, VALIDATION_REPORT_PATH, FEATURE_SUMMARY_PATH, DATE_COL, TARGET_COL
)
from src.data_generation.generator import generate_powergrid_data
from src.validation.validator import run_data_validation, save_validation_report
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.scaler import NumericalScaler
from src.features.temporal import extract_temporal_features
from src.features.engineer import FeatureEngineer, generate_feature_summary
from src.utils.helpers import setup_logger

logger = setup_logger("pipeline")

def split_data_chronologically(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically based on unique dates to avoid lookahead bias.
    
    Parameters:
    df (pd.DataFrame): The dataset containing a Date column
    train_ratio (float): Ratio for training partition
    val_ratio (float): Ratio for validation partition
    
    Returns:
    tuple: (train_df, val_df, test_df)
    """
    logger.info("Performing chronological split of dataset...")
    
    # Temporarily parse dates to sort them correctly
    temp_dates = pd.to_datetime(df[DATE_COL], errors='coerce')
    df_sorted = df.copy()
    df_sorted["_TempDate"] = temp_dates
    # Drop rows with unparseable dates during split (they will be dropped in cleaner anyway)
    df_sorted = df_sorted.dropna(subset=["_TempDate"])
    df_sorted = df_sorted.sort_values(by="_TempDate").reset_index(drop=True)
    
    unique_dates = df_sorted["_TempDate"].unique()
    unique_dates = np.sort(unique_dates)
    
    n_dates = len(unique_dates)
    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))
    
    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]
    
    train_df = df_sorted[df_sorted["_TempDate"].isin(train_dates)].drop(columns=["_TempDate"])
    val_df = df_sorted[df_sorted["_TempDate"].isin(val_dates)].drop(columns=["_TempDate"])
    test_df = df_sorted[df_sorted["_TempDate"].isin(test_dates)].drop(columns=["_TempDate"])
    
    logger.info(f"Split results:")
    logger.info(f"  Train set: {len(train_df)} rows ({len(train_dates)} weeks)")
    logger.info(f"  Val set:   {len(val_df)} rows ({len(val_dates)} weeks)")
    logger.info(f"  Test set:  {len(test_df)} rows ({len(test_dates)} weeks)")
    
    return train_df, val_df, test_df

def run_pipeline(method: str = "ordinal") -> dict:
    """
    Runs the entire end-to-end data preparation pipeline:
    1. Generates synthetic raw dataset (if not already present).
    2. Performs data validation on raw data.
    3. Splits data sequentially (chronological split).
    4. Cleans train, val, and test splits (imputation, outliers, duplicates).
    5. Extracts temporal & cyclical features.
    6. Extracts custom domain and time-series features (lags, rolling averages, metrics).
    7. Encodes categorical columns.
    8. Scales numerical features.
    9. Performs validation on processed datasets to ensure data quality.
    10. Saves final files, summaries, and serialized models.
    """
    logger.info("=========================================")
    logger.info("POWERGRID ML DATA PIPELINE STARTED")
    logger.info("=========================================")
    
    # 1. Load or Generate Raw Data
    if not RAW_DATA_PATH.exists():
        logger.info(f"Raw data not found at {RAW_DATA_PATH}. Generating new dataset...")
        raw_df = generate_powergrid_data(6000)
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"Saved raw data to {RAW_DATA_PATH}")
    else:
        logger.info(f"Loading existing raw data from {RAW_DATA_PATH}...")
        raw_df = pd.read_csv(RAW_DATA_PATH)
        
    # 2. Validate Raw Data
    raw_val_report = run_data_validation(raw_df, stage="raw")
    
    # 3. Chronological Split
    train_df, val_df, test_df = split_data_chronologically(raw_df)
    
    # 4. Clean Splits
    cleaner = DataCleaner()
    cleaner.fit(train_df)
    
    train_clean = cleaner.transform(train_df)
    val_clean = cleaner.transform(val_df)
    test_clean = cleaner.transform(test_df)
    
    # Save the cleaner for future inference
    joblib.dump(cleaner, Path(RAW_DATA_PATH).parent.parent.parent / "models" / "data_cleaner.joblib")
    logger.info("Saved data cleaner model.")
    
    # 5. Extract Temporal Features
    train_temp = extract_temporal_features(train_clean)
    val_temp = extract_temporal_features(val_clean)
    test_temp = extract_temporal_features(test_clean)
    
    # 6. Extract Domain and Time Series Features
    engineer = FeatureEngineer()
    engineer.fit(train_temp)
    
    train_eng = engineer.transform(train_temp)
    val_eng = engineer.transform(val_temp)
    test_eng = engineer.transform(test_temp)
    
    # Save the feature engineer for future inference
    joblib.dump(engineer, Path(RAW_DATA_PATH).parent.parent.parent / "models" / "feature_engineer.joblib")
    logger.info("Saved feature engineer model.")
    
    # Generate feature summary on train set
    feat_summary = generate_feature_summary(train_eng)
    feat_summary.to_csv(FEATURE_SUMMARY_PATH, index=False)
    logger.info(f"Feature summary saved to {FEATURE_SUMMARY_PATH}")
    
    # 7. Encode Categorical Columns
    encoder = CategoricalEncoder(method=method)
    encoder.fit(train_eng)
    
    train_enc = encoder.transform(train_eng)
    val_enc = encoder.transform(val_eng)
    test_enc = encoder.transform(test_eng)
    encoder.save()
    
    # 8. Scale Numerical Features
    scaler = NumericalScaler()
    scaler.fit(train_enc)
    
    train_scaled = scaler.transform(train_enc)
    val_scaled = scaler.transform(val_enc)
    test_scaled = scaler.transform(test_enc)
    scaler.save()
    
    # 9. Validate Cleaned Train Data
    # Validate cleaned train set to confirm that duplicates, nulls, negatives, etc. are eliminated
    processed_val_report = run_data_validation(train_eng, stage="cleaned")
    
    # Combine reports and save
    full_report = pd.concat([raw_val_report, processed_val_report], ignore_index=True)
    save_validation_report(full_report)
    
    # 10. Save processed datasets
    train_scaled.to_csv(PROCESSED_TRAIN_PATH, index=False)
    val_scaled.to_csv(PROCESSED_VAL_PATH, index=False)
    test_scaled.to_csv(PROCESSED_TEST_PATH, index=False)
    
    # Create a concatenated processed dataset as well (requested: processed_dataset.csv)
    # To facilitate unified analysis, combine train/val/test in sequence
    processed_all = pd.concat([train_scaled, val_scaled, test_scaled], ignore_index=True)
    processed_all_path = PROCESSED_DATA_DIR / "processed_dataset.csv"
    processed_all.to_csv(processed_all_path, index=False)
    
    logger.info("=========================================")
    logger.info("POWERGRID ML DATA PIPELINE COMPLETED")
    logger.info("=========================================")
    
    return {
        "raw_shape": raw_df.shape,
        "train_shape": train_scaled.shape,
        "val_shape": val_scaled.shape,
        "test_shape": test_scaled.shape,
        "processed_all_shape": processed_all.shape
    }

if __name__ == "__main__":
    run_pipeline()
