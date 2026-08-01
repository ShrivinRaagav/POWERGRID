import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.config.settings import RESULTS_CSV_PATH, EXPERIMENTS_DIR
from src.utils.helpers import setup_logger
from src.evaluation.statistical_analysis import (
    rank_models, compute_wilcoxon_tests, compute_friedman_test, generate_statistical_report
)
from src.evaluation.publication_plots import generate_all_publication_plots

logger = setup_logger("evaluation_controller")

PRODUCTION_MODELS = ["random_forest", "svr", "xgboost", "mlp", "lstm", "lightgbm_quantile"]

def load_production_predictions(experiments_dir: Path = EXPERIMENTS_DIR) -> Dict[str, pd.DataFrame]:
    """
    Scans the experiments results directory to load the latest predictions CSV 
    for each production model.
    """
    results_root = experiments_dir / "results"
    if not results_root.exists():
        logger.warning(f"Results directory does not exist: {results_root}")
        return {}

    preds_dict: Dict[str, pd.DataFrame] = {}

    # Group runs by production model name
    for model_name in PRODUCTION_MODELS:
        matching_dirs = list(results_root.glob(f"EXP-{model_name}-*"))
        if not matching_dirs:
            continue

        # Sort by modification time or folder name timestamp
        matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_dir = matching_dirs[0]
        preds_file = latest_dir / "predictions.csv"

        if preds_file.exists():
            try:
                df = pd.read_csv(preds_file)
                preds_dict[model_name] = df
                logger.info(f"Loaded predictions for model '{model_name}' from {preds_file.name}")
            except Exception as e:
                logger.error(f"Failed to read predictions for '{model_name}': {e}")

    return preds_dict

def run_full_evaluation(reports_dir: Path = Path("reports")) -> Dict[str, Any]:
    """
    Executes the complete Module 3.5 evaluation pipeline:
    1. Loads metric logs and predictions across all production models.
    2. Builds sorted comparison tables (by RMSE ascending).
    3. Exports model_performance.csv, model_comparison_table.csv, model_summary.md, and best_model.json.
    4. Computes Wilcoxon pairwise tests, Friedman multi-model test, and model rankings.
    5. Exports statistical_evaluation.md, wilcoxon_results.csv, friedman_results.csv, model_ranking.csv.
    6. Generates 10 publication-quality figures in reports/model_plots/.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_csv = Path(RESULTS_CSV_PATH)

    if not results_csv.exists():
        logger.warning(f"Results CSV missing at {results_csv}. Evaluation skipped.")
        return {}

    try:
        df_all = pd.read_csv(results_csv)
    except Exception as e:
        logger.error(f"Failed to read results CSV: {e}")
        return {}

    if df_all.empty:
        logger.warning("Results CSV is empty.")
        return {}

    # Sort chronologically to extract latest run per model
    df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"], errors="coerce")
    df_all = df_all.sort_values(by="Timestamp")

    # Filter to production models only
    latest_df = df_all.groupby("Model Name").last().reset_index()
    prod_df = latest_df[latest_df["Model Name"].isin(PRODUCTION_MODELS)].copy()

    if prod_df.empty:
        logger.warning("No production model runs found in central logs.")
        return {}

    # Compile core metrics table
    report_df = pd.DataFrame()
    report_df["Model"] = prod_df["Model Name"]
    report_df["MAE"] = pd.to_numeric(prod_df["MAE"], errors="coerce")
    report_df["RMSE"] = pd.to_numeric(prod_df["RMSE"], errors="coerce")
    report_df["MAPE"] = pd.to_numeric(prod_df["MAPE"], errors="coerce")
    report_df["WMAPE"] = pd.to_numeric(prod_df["WMAPE"] if "WMAPE" in prod_df.columns else pd.Series([np.nan] * len(prod_df)), errors="coerce")
    report_df["SMAPE"] = pd.to_numeric(prod_df["SMAPE"], errors="coerce")
    report_df["R²"] = pd.to_numeric(prod_df["R2"], errors="coerce")
    report_df["Training Time"] = pd.to_numeric(prod_df["Training Time"], errors="coerce")
    report_df["Inference Time"] = pd.to_numeric(prod_df["Inference Time"], errors="coerce")

    # Parse Pinball Loss (only reported for lightgbm_quantile)
    pinball_col = []
    for _, row in prod_df.iterrows():
        if row["Model Name"] == "lightgbm_quantile":
            loss_str = row.get("Pinball Loss", "N/A")
            if pd.notnull(loss_str) and loss_str not in ["N/A", ""]:
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

    # Sort strictly by RMSE ascending
    report_df = report_df.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    # 1. Save model_performance.csv and model_comparison_table.csv
    export_df = report_df.copy()
    export_df["WMAPE"] = export_df["WMAPE"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    export_df.to_csv(reports_dir / "model_performance.csv", index=False)
    export_df.to_csv(reports_dir / "model_comparison_table.csv", index=False)

    # 2. Select Best Model
    best_row = report_df.iloc[0]
    best_model_name = str(best_row["Model"])

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

    # 3. Generate model_summary.md
    summary_md = f"""# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented production forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | WMAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in report_df.iterrows():
        mae_v = f"{row['MAE']:.6f}" if pd.notnull(row['MAE']) else "N/A"
        rmse_v = f"{row['RMSE']:.6f}" if pd.notnull(row['RMSE']) else "N/A"
        mape_v = f"{row['MAPE']:.4f}%" if pd.notnull(row['MAPE']) else "N/A"
        wmape_v = f"{row['WMAPE']:.2f}%" if pd.notnull(row['WMAPE']) else "N/A"
        smape_v = f"{row['SMAPE']:.4f}%" if pd.notnull(row['SMAPE']) else "N/A"
        r2_v = f"{row['R²']:.6f}" if pd.notnull(row['R²']) else "N/A"
        t_time = f"{row['Training Time']:.4f}" if pd.notnull(row['Training Time']) else "N/A"
        i_time = f"{row['Inference Time']:.4f}" if pd.notnull(row['Inference Time']) else "N/A"
        pin_v = str(row["Pinball Loss"])
        summary_md += f"| {row['Model']} | {mae_v} | {rmse_v} | {mape_v} | {wmape_v} | {smape_v} | {r2_v} | {t_time} | {i_time} | {pin_v} |\n"

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
- **MAPE**: Average percentage prediction error.
- **WMAPE**: Weighted Mean Absolute Percentage Error. Recommended for demand forecasting.
- **SMAPE**: Symmetric percentage error between actual and predicted values.
- **R²**: Coefficient of determination measuring explained variance.
- **Pinball Loss**: Measures quality of probabilistic quantile predictions (LightGBM Quantile only).
"""
    with open(reports_dir / "model_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    # 4. Perform Statistical Analysis & Model Ranking
    preds_dict = load_production_predictions()
    ranking_df = rank_models(report_df)
    ranking_df.to_csv(reports_dir / "model_ranking.csv", index=False)

    wilcoxon_df = compute_wilcoxon_tests(preds_dict, alpha=0.05)
    wilcoxon_df.to_csv(reports_dir / "wilcoxon_results.csv", index=False)

    friedman_meta, friedman_df = compute_friedman_test(preds_dict, alpha=0.05)
    friedman_df.to_csv(reports_dir / "friedman_results.csv", index=False)

    generate_statistical_report(ranking_df, wilcoxon_df, friedman_meta, reports_dir / "statistical_evaluation.md")

    # 5. Generate IEEE Publication Plots (10 figures)
    best_preds_df = preds_dict.get(best_model_name, None)
    quantile_preds_df = preds_dict.get("lightgbm_quantile", None)

    generate_all_publication_plots(
        report_df=report_df,
        best_preds_df=best_preds_df,
        quantile_preds_df=quantile_preds_df,
        output_dir=reports_dir / "model_plots",
        best_model_name=best_model_name,
        dpi=300
    )

    # 6. Run Module 4 (SHAP Explainability) automatically
    try:
        from src.explainability.run_explainability import run_explainability_pipeline
        run_explainability_pipeline(reports_dir=reports_dir)
    except Exception as e:
        logger.error(f"Module 4 SHAP explainability pipeline failed: {e}", exc_info=True)

    logger.info("Module 3.5 & Module 4 evaluation & explainability pipelines completed successfully.")
    return {
        "best_model": best_model_name,
        "ranking_df": ranking_df,
        "wilcoxon_df": wilcoxon_df,
        "friedman_meta": friedman_meta
    }

def main():
    parser = argparse.ArgumentParser(description="POWERGRID Forecast Evaluation & Statistical Controller (Module 3.5)")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save evaluation reports.")
    args = parser.parse_args()

    run_full_evaluation(reports_dir=Path(args.reports_dir))

if __name__ == "__main__":
    main()
