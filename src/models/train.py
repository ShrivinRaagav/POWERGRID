import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path

from src.config.settings import (
    PROCESSED_DATASET_PATH, MODELS_CONFIG, FORECAST_RANDOM_SEED, DATE_COL, TARGET_COL
)
from src.models.registry import get_model_class, list_registered_models
# Import mock_model and other models so they register automatically
import src.models.mock_model
import src.models.random_forest
import src.models.svr
import src.models.xgboost_model
import src.models.mlp
import src.models.lstm
import src.models.lightgbm_quantile
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
    if isinstance(preds, dict):
        df_preds["Forecast_Prediction_P10"] = preds["P10"]
        df_preds["Forecast_Prediction"] = preds["P50"]
        df_preds["Forecast_Prediction_P90"] = preds["P90"]
    else:
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
    
    # 12. Automatically update reports
    update_reports()

def update_reports():
    """Compiles results from all runs to generate performance reports and best model selection metadata."""
    import json
    from datetime import datetime
    import src.config.settings as settings
    
    results_path = Path(settings.RESULTS_CSV_PATH)
    if not results_path.exists():
        logger.warning(f"Results CSV not found at {results_path}. Skipping reports update.")
        return
        
    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        logger.error(f"Failed to read results CSV: {e}")
        return
        
    if df.empty:
        logger.warning("Results CSV is empty. Skipping reports update.")
        return
        
    # Convert Timestamp to datetime to find the latest runs
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values(by="Timestamp")
    
    # Keep only the latest run for each unique model
    latest_df = df.groupby("Model Name").last().reset_index()
    
    # Filter only production models, exclude mock/test models
    production_models = ["random_forest", "svr", "xgboost", "mlp", "lstm", "lightgbm_quantile"]
    latest_df = latest_df[latest_df["Model Name"].isin(production_models)]
    if latest_df.empty:
        logger.warning("No production model runs found in results. Skipping reports update.")
        return
        
    report_df = pd.DataFrame()
    report_df["Model"] = latest_df["Model Name"]
    report_df["MAE"] = pd.to_numeric(latest_df["MAE"], errors="coerce")
    report_df["RMSE"] = pd.to_numeric(latest_df["RMSE"], errors="coerce")
    report_df["MAPE"] = pd.to_numeric(latest_df["MAPE"], errors="coerce")
    report_df["WMAPE"] = pd.to_numeric(latest_df["WMAPE"] if "WMAPE" in latest_df.columns else pd.Series([np.nan]*len(latest_df)), errors="coerce")
    report_df["SMAPE"] = pd.to_numeric(latest_df["SMAPE"], errors="coerce")
    report_df["R²"] = pd.to_numeric(latest_df["R2"], errors="coerce")
    report_df["Training Time"] = pd.to_numeric(latest_df["Training Time"], errors="coerce")
    report_df["Inference Time"] = pd.to_numeric(latest_df["Inference Time"], errors="coerce")
    
    # Extract Pinball Loss for LightGBM Quantile Regression
    pinball_col = []
    for _, row in latest_df.iterrows():
        if row["Model Name"] == "lightgbm_quantile":
            loss_str = row.get("Pinball Loss", "N/A")
            if pd.notnull(loss_str) and loss_str != "N/A" and loss_str != "":
                try:
                    losses = json.loads(loss_str)
                    val_10 = losses.get("Pinball_Loss_0.10", losses.get("Pinball_Loss_0.1", None))
                    val_50 = losses.get("Pinball_Loss_0.50", losses.get("Pinball_Loss_0.5", None))
                    val_90 = losses.get("Pinball_Loss_0.90", losses.get("Pinball_Loss_0.9", None))
                    if val_10 is not None and val_50 is not None and val_90 is not None:
                        avg_val = (float(val_10) + float(val_50) + float(val_90)) / 3.0
                        pinball_col.append(f"{avg_val:.6f}")
                    else:
                        pinball_col.append(loss_str)
                except Exception:
                    pinball_col.append(loss_str)
            else:
                pinball_col.append("N/A")
        else:
            pinball_col.append("N/A")
            
    report_df["Pinball Loss"] = pinball_col
    
    # Sort the comparison table by RMSE in ascending order (best performing model first)
    report_df = report_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)
    
    # Save reports
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Export with WMAPE formatted as a percentage string
    export_df = report_df.copy()
    export_df["WMAPE"] = export_df["WMAPE"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    
    export_df.to_csv(reports_dir / "model_performance.csv", index=False)
    export_df.to_csv(reports_dir / "model_comparison_table.csv", index=False)
    
    # Select best model (sorted first row)
    best_row = report_df.iloc[0]
    best_model_name = best_row["Model"]
    
    best_metrics = {
        "MAE": float(best_row["MAE"]) if pd.notnull(best_row["MAE"]) else None,
        "RMSE": float(best_row["RMSE"]) if pd.notnull(best_row["RMSE"]) else None,
        "MAPE": float(best_row["MAPE"]) if pd.notnull(best_row["MAPE"]) else None,
        "WMAPE": f"{float(best_row['WMAPE']):.2f}%" if pd.notnull(best_row["WMAPE"]) else None,
        "SMAPE": float(best_row["SMAPE"]) if pd.notnull(best_row["SMAPE"]) else None,
        "R2": float(best_row["R²"]) if pd.notnull(best_row["R²"]) else None,
        "PinballLoss": best_row["Pinball Loss"] if best_row["Pinball Loss"] != "N/A" else None
    }
    
    best_model_meta = {
        "best_model": best_model_name,
        "selection_criterion": "Lowest RMSE",
        "evaluation_metrics": best_metrics,
        "generated_at": datetime.now().isoformat()
    }
    
    with open(reports_dir / "best_model.json", "w", encoding="utf-8") as f:
        json.dump(best_model_meta, f, indent=4)
        
    # Generate model_summary.md
    summary_md = f"""# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented production forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | WMAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in report_df.iterrows():
        mae_val = f"{row['MAE']:.6f}" if pd.notnull(row['MAE']) else "N/A"
        rmse_val = f"{row['RMSE']:.6f}" if pd.notnull(row['RMSE']) else "N/A"
        mape_val = f"{row['MAPE']:.4f}%" if pd.notnull(row['MAPE']) else "N/A"
        wmape_val = f"{row['WMAPE']:.2f}%" if pd.notnull(row['WMAPE']) else "N/A"
        smape_val = f"{row['SMAPE']:.4f}%" if pd.notnull(row['SMAPE']) else "N/A"
        r2_val = f"{row['R²']:.6f}" if pd.notnull(row['R²']) else "N/A"
        t_time = f"{row['Training Time']:.4f}" if pd.notnull(row['Training Time']) else "N/A"
        i_time = f"{row['Inference Time']:.4f}" if pd.notnull(row['Inference Time']) else "N/A"
        pin_val = str(row["Pinball Loss"])
        
        summary_md += f"| {row['Model']} | {mae_val} | {rmse_val} | {mape_val} | {wmape_val} | {smape_val} | {r2_val} | {t_time} | {i_time} | {pin_val} |\n"
        
    summary_md += f"""
---

## Best Performing Model

Model:
{best_model_name}

Selection Criterion:
Lowest RMSE on the held-out chronological test dataset.

Performance Summary:
- RMSE: {best_row['RMSE']:.6f}
- MAE: {best_row['MAE']:.6f}
- WMAPE: {best_row['WMAPE']:.2f}%
- R²: {best_row['R²']:.6f}

Future Usage:
This model will be forwarded to:
- Module 3.5 (Forecast Model Evaluation)
- Module 4 (SHAP Explainability)
- Module 5 (Multi-Objective Supply Chain Optimization)

---

## Metric Documentation

- **MAE**: Average absolute prediction error.
- **RMSE**: Penalizes larger prediction errors more heavily.
- **MAPE**: Average percentage prediction error. Computed using a numerically stable implementation that ignores zero or near-zero targets.
- **WMAPE**: Weighted Mean Absolute Percentage Error. Recommended for demand forecasting because it remains stable even when demand contains small values.
- **SMAPE**: Symmetric percentage error between actual and predicted values.
- **R²**: Coefficient of determination measuring explained variance.
- **Pinball Loss**: Measures the quality of probabilistic (quantile) predictions and is only applicable to LightGBM Quantile Regression.

---

> [!NOTE]
> All evaluation metrics reported above are computed on the independent chronological test dataset to ensure an unbiased comparison.
"""
    with open(reports_dir / "model_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)
        
    logger.info("Forecasting reports successfully generated.")

def main():
    parser = argparse.ArgumentParser(description="POWERGRID Demand Forecasting Controller CLI (Module 3)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list_registered_models() + ["random_forest", "xgboost", "lightgbm_quantile", "lstm", "mlp", "svr", "prophet"],
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
