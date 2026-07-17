import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path

from src.config.settings import (
    PROCESSED_DATASET_PATH, MODELS_CONFIG, FORECAST_RANDOM_SEED, DATE_COL, TARGET_COL
)
from src.models.registry import get_model_class, list_registered_models
# Import mock_model so it registers itself automatically
import src.models.mock_model
from src.experiments.experiment_manager import ExperimentManager
from src.experiments.experiment_logger import log_experiment_run
from src.utils.helpers import setup_logger

logger = setup_logger("train_controller")

def split_chronological(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the panel/time-series dataset chronologically."""
    df_sorted = df.sort_values(by=[DATE_COL]).copy()
    unique_dates = df_sorted[DATE_COL].unique()
    n_dates = len(unique_dates)
    
    n_train = int(n_dates * train_ratio)
    n_val = int(n_dates * val_ratio)
    
    train_dates = unique_dates[:n_train]
    val_dates = unique_dates[n_train:n_train + n_val]
    test_dates = unique_dates[n_train + n_val:]
    
    train_df = df_sorted[df_sorted[DATE_COL].isin(train_dates)]
    val_df = df_sorted[df_sorted[DATE_COL].isin(val_dates)]
    test_df = df_sorted[df_sorted[DATE_COL].isin(test_dates)]
    
    return train_df, val_df, test_df

def run_training_flow(model_name: str, notes: str = ""):
    """Orchestrates the model instantiation, training, prediction, evaluation, and logging."""
    logger.info(f"Initiating training controller flow for model: '{model_name}'...")
    
    # 1. Retrieve the model class from the registry
    try:
        model_class = get_model_class(model_name)
    except KeyError as e:
        logger.error(f"Error loading model class: {e}")
        return
        
    # 2. Load hyperparameters from configuration
    hyperparams = MODELS_CONFIG.get(model_name, {})
    logger.info(f"Loaded hyperparameters: {hyperparams}")
    
    # 3. Load dataset
    if not PROCESSED_DATASET_PATH.exists():
        logger.error(f"Processed dataset not found at {PROCESSED_DATASET_PATH}. Run the preprocessing pipeline first.")
        return
        
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    logger.info(f"Loaded dataset of shape: {df.shape}")
    
    # 4. Split chronologically
    train_df, val_df, test_df = split_chronological(df)
    logger.info(f"Chronological split: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")
    
    # Extract feature list (all columns except Date, Project_ID, Region, Material_Type, and Target)
    id_cols = [c for c in ["Date", "Project_ID", "Region", "Material_Type"] if c in df.columns]
    feature_cols = [col for col in df.columns if col not in id_cols and col != TARGET_COL]
    
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL]
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL]
    
    # 5. Initialize model
    model = model_class(**hyperparams)
    
    # 6. Fit model (measure training time)
    logger.info("Training started...")
    start_train = time.perf_counter()
    model.train(X_train, y_train, X_val, y_val)
    train_time = time.perf_counter() - start_train
    logger.info(f"Training completed in {train_time:.4f}s.")
    
    # 7. Generate predictions on test set (measure inference time)
    logger.info("Generating forecasts on test set...")
    start_inf = time.perf_counter()
    preds = model.predict(X_test)
    inference_time = time.perf_counter() - start_inf
    logger.info(f"Inference completed in {inference_time:.4f}s.")
    
    # 8. Evaluate metrics
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"Evaluation Metrics: {metrics}")
    
    # 9. Initialize Experiment Manager to structure directories
    manager = ExperimentManager(model_name=model_name)
    
    # 10. Save checkpoints and predictions
    manager.save_model_checkpoint(model, filename="model.joblib")
    
    # Save predictions alongside actuals
    df_preds = test_df[id_cols + [TARGET_COL]].copy()
    df_preds["Forecast_Prediction"] = preds
    manager.save_predictions(df_preds, filename="predictions.csv")
    
    # 11. Log results to the central CSV file
    log_experiment_run(
        model_name=model_name,
        hyperparams=hyperparams,
        train_time=train_time,
        inference_time=inference_time,
        metrics=metrics,
        notes=notes
    )
    
    logger.info(f"Forecasting pipeline completed successfully for run: {manager.run_id}")

def main():
    parser = argparse.ArgumentParser(description="POWERGRID Demand Forecasting Controller CLI (Module 3)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list_registered_models() + ["random_forest", "xgboost", "lightgbm", "lstm", "mlp", "svr", "prophet"],
        help="Target forecasting model name to run."
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes describing the experiment run."
    )
    args = parser.parse_args()
    
    # If the user selects a model that is configured but not yet registered, notify them.
    registered = list_registered_models()
    model_lower = args.model.strip().lower()
    
    if model_lower not in registered:
        print(f"\n[ERROR] Model '{args.model}' is configured but not yet implemented/registered in the codebase.")
        print(f"Registered models currently available: {registered}\n")
        return
        
    run_training_flow(model_name=model_lower, notes=args.notes)

if __name__ == "__main__":
    main()
