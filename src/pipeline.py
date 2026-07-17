import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
import joblib

from src.config.settings import (
    RAW_DATA_PATH, PROCESSED_TRAIN_PATH, PROCESSED_VAL_PATH, PROCESSED_TEST_PATH,
    PROCESSED_DATASET_PATH, VALIDATION_REPORT_PATH, FEATURE_SUMMARY_PATH, 
    DATA_QUALITY_REPORT_PATH, DATA_DICTIONARY_PATH, PIPELINE_DIAGRAM_PATH,
    DATE_COL, TARGET_COL, TRAIN_RATIO, VAL_RATIO
)
from src.data_generation.generator import generate_powergrid_data
from src.validation.validator import run_data_validation, save_validation_report
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.scaler import NumericalScaler
from src.features.temporal import extract_temporal_features
from src.features.engineer import FeatureEngineer, generate_feature_summary
from src.utils.helpers import setup_logger
from src.utils.reports_generator import (
    generate_data_quality_report, generate_data_dictionary, generate_pipeline_diagram
)
from src.time_series.decomposition import TimeSeriesFeatureExtractor
from src.time_series.visualization import generate_important_plots
from src.time_series.reconstruction_validator import run_reconstruction_validation
from src.feature_selection.feature_selector import FeatureSelector
from src.utils.catalog_generator import generate_feature_catalog
from src.config.settings import (
    FS_RECONSTRUCTION_REPORT_PATH, FS_QUALITY_REPORT_PATH, FS_CATALOG_PATH,
    DATASET_VERSION_PATH, FORECAST_RANDOM_SEED
)
from src.utils.version_writer import generate_dataset_version_report




logger = setup_logger("pipeline")

def split_data_chronologically(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically based on unique dates to avoid lookahead bias.
    """
    logger.info("Performing chronological split of dataset...")
    
    # Temporarily parse dates to sort them correctly
    temp_dates = pd.to_datetime(df[DATE_COL], errors='coerce')
    df_sorted = df.copy()
    df_sorted["_TempDate"] = temp_dates
    # Drop rows with unparseable dates during split
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
    pipeline_start = time.perf_counter()
    logger.info("=========================================")
    logger.info("POWERGRID ML DATA PIPELINE STARTED")
    logger.info("=========================================")
    
    metrics = {}
    
    # 1. Load or Generate Raw Data
    stage_start = time.perf_counter()
    if not RAW_DATA_PATH.exists():
        logger.info(f"Raw data not found at {RAW_DATA_PATH}. Generating new dataset...")
        raw_df = generate_powergrid_data(6000)
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"Saved raw data to {RAW_DATA_PATH}")
    else:
        logger.info(f"Loading existing raw data from {RAW_DATA_PATH}...")
        raw_df = pd.read_csv(RAW_DATA_PATH)
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Data Generation / Load' completed in {duration:.4f}s. Records: {len(raw_df)}")
    metrics["load"] = {"start": stage_start, "duration": duration, "records": len(raw_df)}
    
    # Pre-calculate anomalies count for reporting
    raw_missing_count = int(raw_df.isnull().sum().sum())
    raw_dup_count = int(raw_df.duplicated().sum())

    # 2. Validate Raw Data
    stage_start = time.perf_counter()
    raw_val_report = run_data_validation(raw_df, stage="raw")
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Raw Validation' completed in {duration:.4f}s.")
    metrics["val_raw"] = {"start": stage_start, "duration": duration}
    
    # 3. Chronological Split
    stage_start = time.perf_counter()
    train_df, val_df, test_df = split_data_chronologically(raw_df)
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Chronological Split' completed in {duration:.4f}s.")
    metrics["split"] = {"start": stage_start, "duration": duration}
    
    # 4. Clean Splits
    stage_start = time.perf_counter()
    cleaner = DataCleaner()
    cleaner.fit(train_df)
    
    train_clean = cleaner.transform(train_df)
    val_clean = cleaner.transform(val_df)
    test_clean = cleaner.transform(test_df)
    
    # Save the cleaner for future inference
    joblib.dump(cleaner, Path(RAW_DATA_PATH).parent.parent.parent / "models" / "data_cleaner.joblib")
    duration = time.perf_counter() - stage_start
    
    records_removed = len(raw_df) - (len(train_clean) + len(val_clean) + len(test_clean))
    logger.info(f"Stage 'Data Cleaning' completed in {duration:.4f}s. Records removed: {records_removed}")
    metrics["clean"] = {"start": stage_start, "duration": duration, "removed": records_removed}
    
    # 5. Extract Temporal Features
    stage_start = time.perf_counter()
    train_temp = extract_temporal_features(train_clean)
    val_temp = extract_temporal_features(val_clean)
    test_temp = extract_temporal_features(test_clean)
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Temporal Feature Extraction' completed in {duration:.4f}s.")
    metrics["temporal"] = {"start": stage_start, "duration": duration}
    
    # 6. Extract Domain and Time Series Features
    stage_start = time.perf_counter()
    engineer = FeatureEngineer()
    engineer.fit(train_temp)
    
    train_eng = engineer.transform(train_temp)
    val_eng = engineer.transform(val_temp)
    test_eng = engineer.transform(test_temp)
    
    # Save the feature engineer for future inference
    joblib.dump(engineer, Path(RAW_DATA_PATH).parent.parent.parent / "models" / "feature_engineer.joblib")
    
    # Generate feature summary on train set
    feat_summary = generate_feature_summary(train_eng)
    feat_summary.to_csv(FEATURE_SUMMARY_PATH, index=False)
    duration = time.perf_counter() - stage_start
    
    features_created = len(train_eng.columns) - len(train_temp.columns)
    logger.info(f"Stage 'Domain Feature Engineering' completed in {duration:.4f}s. Features created: {features_created}")
    metrics["engineer"] = {"start": stage_start, "duration": duration, "features_created": features_created}
    
    # 7. Encode Categorical Columns
    stage_start = time.perf_counter()
    encoder = CategoricalEncoder(method=method)
    encoder.fit(train_eng)
    
    train_enc = encoder.transform(train_eng)
    val_enc = encoder.transform(val_eng)
    test_enc = encoder.transform(test_eng)
    encoder.save()
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Categorical Encoding' completed in {duration:.4f}s.")
    metrics["encode"] = {"start": stage_start, "duration": duration}
    
    # 8. Scale Numerical Features
    stage_start = time.perf_counter()
    scaler = NumericalScaler()
    scaler.fit(train_enc)
    
    train_scaled = scaler.transform(train_enc)
    val_scaled = scaler.transform(val_enc)
    test_scaled = scaler.transform(test_enc)
    scaler.save()
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Numerical Scaling' completed in {duration:.4f}s.")
    metrics["scale"] = {"start": stage_start, "duration": duration}
    
    # 9. Validate Cleaned Train Data
    stage_start = time.perf_counter()
    processed_val_report = run_data_validation(train_eng, stage="cleaned")
    
    # Combine validation reports and save
    full_report = pd.concat([raw_val_report, processed_val_report], ignore_index=True)
    save_validation_report(full_report)
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Clean Validation' completed in {duration:.4f}s.")
    metrics["val_clean"] = {"start": stage_start, "duration": duration}
    
    # 10. Save processed datasets
    train_scaled.to_csv(PROCESSED_TRAIN_PATH, index=False)
    val_scaled.to_csv(PROCESSED_VAL_PATH, index=False)
    test_scaled.to_csv(PROCESSED_TEST_PATH, index=False)
    
    # Create combined processed dataset (processed_dataset.csv)
    processed_all = pd.concat([train_scaled, val_scaled, test_scaled], ignore_index=True)
    processed_all.to_csv(PROCESSED_DATASET_PATH, index=False)
    
    # 10.5. Run Time-Series Decomposition & Feature Extraction (Module 2)
    stage_start = time.perf_counter()
    logger.info("Running Time-Series Decomposition & Feature Extraction (Module 2)...")
    extractor = TimeSeriesFeatureExtractor()
    processed_enriched, df_summary = extractor.fit_transform(processed_all)
    
    # Run Reconstruction Validation & Decomposition Quality report
    run_reconstruction_validation(
        df_ts=processed_enriched,
        group_cols=extractor.group_cols,
        date_col=extractor.date_col,
        target_col=extractor.target_col,
        dwt_transformer=extractor.dwt_transformer,
        emd_processor=extractor.emd_processor,
        csv_report_path=FS_RECONSTRUCTION_REPORT_PATH,
        md_report_path=FS_QUALITY_REPORT_PATH
    )
    
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Time-Series Decomposition' completed in {duration:.4f}s.")
    metrics["time_series_decomp"] = {"start": stage_start, "duration": duration}

    # 10.6. Run Feature Selection & Dimensionality Reduction
    stage_start = time.perf_counter()
    logger.info("Running Feature Selection & Dimensionality Reduction...")
    selector = FeatureSelector()
    selector.fit(processed_enriched, target_col=extractor.target_col)
    processed_selected = selector.transform(processed_enriched)
    
    # Overwrite combined processed dataset with final optimized selected columns
    processed_selected.to_csv(PROCESSED_DATASET_PATH, index=False)
    
    # Generate dataset version report
    generate_dataset_version_report(
        df=processed_selected,
        random_seed=FORECAST_RANDOM_SEED,
        output_path=DATASET_VERSION_PATH
    )
    
    # Generate dynamic feature catalog
    generate_feature_catalog(
        all_features=list(processed_enriched.columns),
        selected_features=list(processed_selected.columns),
        output_path=FS_CATALOG_PATH
    )
    
    # Generate publication-quality important plots
    generate_important_plots(
        df_enriched=processed_enriched,
        df_selected=processed_selected,
        feature_selector=selector,
        dwt_transformer=extractor.dwt_transformer,
        emd_processor=extractor.emd_processor,
        group_cols=extractor.group_cols,
        date_col=extractor.date_col,
        target_col=extractor.target_col
    )
    
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Feature Selection & Dimensionality Reduction' completed in {duration:.4f}s.")
    metrics["feature_selection"] = {"start": stage_start, "duration": duration}
    
    # 11. Generate Markdown Research Reports
    stage_start = time.perf_counter()
    logger.info("Generating markdown reports (Data Quality, Data Dictionary, Pipeline Diagram)...")
    
    # Generate Data Quality Report
    generate_data_quality_report(
        raw_df=raw_df,
        cleaned_df=train_eng,
        validation_df=full_report,
        feat_summary_df=feat_summary,
        output_path=DATA_QUALITY_REPORT_PATH
    )
    # Generate Data Dictionary
    generate_data_dictionary(
        df=train_eng,
        output_path=DATA_DICTIONARY_PATH
    )
    # Generate Pipeline Diagram
    generate_pipeline_diagram(
        output_path=PIPELINE_DIAGRAM_PATH
    )
    
    duration = time.perf_counter() - stage_start
    logger.info(f"Stage 'Reports Generation' completed in {duration:.4f}s.")
    metrics["reports"] = {"start": stage_start, "duration": duration}
    
    pipeline_duration = time.perf_counter() - pipeline_start
    logger.info("=========================================")
    logger.info(f"POWERGRID ML DATA PIPELINE COMPLETED IN {pipeline_duration:.4f}s")
    logger.info("=========================================")
    
    return {
        "raw_shape": raw_df.shape,
        "train_shape": train_scaled.shape,
        "val_shape": val_scaled.shape,
        "test_shape": test_scaled.shape,
        "processed_all_shape": processed_selected.shape,
        "duration": pipeline_duration,
        "raw_anomalies": {
            "missing": raw_missing_count,
            "duplicates": raw_dup_count
        }
    }

if __name__ == "__main__":
    run_pipeline()
