import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.helpers import setup_logger

logger = setup_logger("publication_plots")

def apply_ieee_style():
    """Sets global matplotlib parameters for IEEE publication quality standards."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.edgecolor': 'black',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.family': 'sans-serif',
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.autolayout': True
    })

def _plot_metric_bar(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    y_label: str,
    save_path: Path,
    ascending: bool = True,
    is_percentage: bool = False,
    dpi: int = 300
):
    """Generic helper function to generate IEEE-quality metric bar comparisons."""
    apply_ieee_style()
    if df.empty or metric_col not in df.columns:
        logger.warning(f"Metric '{metric_col}' not available for plotting.")
        return

    plot_df = df.copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col].astype(str).str.replace("%", ""), errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(by=metric_col, ascending=ascending)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')
    models = plot_df["Model"].astype(str).tolist()
    values = plot_df[metric_col].values

    # Color palette
    colors = ["#1f77b4", "#33a02c", "#ff7f0e", "#e31a1c", "#9467bd", "#8c564b", "#e377c2"]
    bar_colors = [colors[i % len(colors)] for i in range(len(models))]

    bars = ax.bar(models, values, color=bar_colors, edgecolor="black", linewidth=1.0, alpha=0.85, width=0.55)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Forecasting Model")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.5, color="gray", axis="y")
    ax.set_axisbelow(True)

    # Value labels on top of bars
    y_max = np.max(values) if len(values) > 0 else 1.0
    ax.set_ylim(0, y_max * 1.18)

    for bar in bars:
        height = bar.get_height()
        val_str = f"{height:.2f}%" if is_percentage else f"{height:.4f}"
        ax.annotate(
            val_str,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.xticks(rotation=15, ha="right")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path.resolve()), dpi=dpi, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    logger.info(f"Saved publication plot to {save_path}")

def plot_actual_vs_predicted_best(
    best_preds_df: pd.DataFrame,
    save_path: Path,
    best_model_name: str = "Best Model",
    target_col: str = "Quantity_Required",
    pred_col: str = "Forecast_Prediction",
    dpi: int = 300
):
    """Plots IEEE-quality actual vs predicted demand sequence and scatter for the top model."""
    apply_ieee_style()
    if best_preds_df.empty or target_col not in best_preds_df.columns or pred_col not in best_preds_df.columns:
        logger.warning("Invalid predictions DataFrame for best model plot.")
        return

    y_t = best_preds_df[target_col].values
    y_p = best_preds_df[pred_col].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

    # Subplot 1: Timeline Comparison
    axes[0].plot(y_t, label="Actual Demand", color="#1f77b4", linewidth=1.6)
    axes[0].plot(y_p, label=f"Predicted ({best_model_name})", color="#d62728", linewidth=1.4, linestyle="--")
    axes[0].set_title(f"Demand Timeline Comparison ({best_model_name})")
    axes[0].set_xlabel("Time Index (Test Set)")
    axes[0].set_ylabel("Material Demand Quantity")
    axes[0].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black")
    axes[0].grid(True, linestyle="--", alpha=0.5, color="gray")

    # Subplot 2: Scatter Plot with Parity Line
    axes[1].scatter(y_t, y_p, color="#2ca02c", alpha=0.6, edgecolors="none", s=25, label="Test Samples")
    min_val = min(np.min(y_t), np.min(y_p))
    max_val = max(np.max(y_t), np.max(y_p))
    axes[1].plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1.5, label="Perfect Parity (y = x)")
    axes[1].set_title("Actual vs Predicted Scatter")
    axes[1].set_xlabel("Actual Demand")
    axes[1].set_ylabel("Predicted Demand")
    axes[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="black")
    axes[1].grid(True, linestyle="--", alpha=0.5, color="gray")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path.resolve()), dpi=dpi, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    logger.info(f"Saved best model actual vs predicted plot to {save_path}")

def plot_probabilistic_forecast(
    quantile_df: pd.DataFrame,
    save_path: Path,
    target_col: str = "Quantity_Required",
    p10_col: str = "Forecast_Prediction_P10",
    p50_col: str = "Forecast_Prediction",
    p90_col: str = "Forecast_Prediction_P90",
    dpi: int = 300
):
    """Plots IEEE-quality probabilistic forecast intervals (P10, P50, P90) for LightGBM Quantile model."""
    apply_ieee_style()
    req_cols = [target_col, p10_col, p50_col, p90_col]
    missing = [c for c in req_cols if c not in quantile_df.columns]
    if missing:
        logger.warning(f"Missing columns for probabilistic forecast plot: {missing}")
        return

    y_t = quantile_df[target_col].values
    y_10 = quantile_df[p10_col].values
    y_50 = quantile_df[p50_col].values
    y_90 = quantile_df[p90_col].values
    t = np.arange(len(y_t))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')

    # Shaded confidence band (P10 - P90)
    ax.fill_between(t, y_10, y_90, color="#1f77b4", alpha=0.25, label="80% Prediction Interval (P10 - P90)")

    # Demand lines
    ax.plot(t, y_t, color="black", linewidth=1.6, label="Actual Demand")
    ax.plot(t, y_50, color="#d62728", linewidth=1.5, linestyle="-", label="Median Forecast (P50)")
    ax.plot(t, y_10, color="#1f77b4", linewidth=1.0, linestyle="--", alpha=0.8, label="P10 Lower Bound")
    ax.plot(t, y_90, color="#1f77b4", linewidth=1.0, linestyle="--", alpha=0.8, label="P90 Upper Bound")

    ax.set_title("Probabilistic Demand Forecast (LightGBM Quantile Regression)")
    ax.set_xlabel("Time Index (Test Set)")
    ax.set_ylabel("Material Demand Quantity")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black")
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    logger.info(f"Saved probabilistic forecast plot to {save_path}")

def generate_all_publication_plots(
    report_df: pd.DataFrame,
    best_preds_df: Optional[pd.DataFrame],
    quantile_preds_df: Optional[pd.DataFrame],
    output_dir: Path = Path("reports/model_plots"),
    best_model_name: str = "Best Model",
    dpi: int = 300
):
    """
    Generates all 10 IEEE-quality publication plots in reports/model_plots/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. MAE Comparison
    _plot_metric_bar(report_df, "MAE", "Mean Absolute Error (MAE) Comparison", "MAE (Lower is Better)", output_dir / "mae_comparison.png", ascending=True, dpi=dpi)

    # 2. RMSE Comparison
    _plot_metric_bar(report_df, "RMSE", "Root Mean Squared Error (RMSE) Comparison", "RMSE (Lower is Better)", output_dir / "rmse_comparison.png", ascending=True, dpi=dpi)

    # 3. MAPE Comparison
    _plot_metric_bar(report_df, "MAPE", "Mean Absolute Percentage Error (MAPE) Comparison", "MAPE % (Lower is Better)", output_dir / "mape_comparison.png", ascending=True, is_percentage=True, dpi=dpi)

    # 4. WMAPE Comparison
    _plot_metric_bar(report_df, "WMAPE", "Weighted MAPE (WMAPE) Comparison", "WMAPE % (Lower is Better)", output_dir / "wmape_comparison.png", ascending=True, is_percentage=True, dpi=dpi)

    # 5. SMAPE Comparison
    _plot_metric_bar(report_df, "SMAPE", "Symmetric MAPE (SMAPE) Comparison", "SMAPE % (Lower is Better)", output_dir / "smape_comparison.png", ascending=True, is_percentage=True, dpi=dpi)

    # 6. R² Comparison
    _plot_metric_bar(report_df, "R²", "R² Coefficient of Determination Comparison", "R² Score (Higher is Better)", output_dir / "r2_comparison.png", ascending=False, dpi=dpi)

    # 7. Training Time Comparison
    _plot_metric_bar(report_df, "Training Time", "Training Time Comparison", "Time (seconds)", output_dir / "training_time_comparison.png", ascending=True, dpi=dpi)

    # 8. Inference Time Comparison
    _plot_metric_bar(report_df, "Inference Time", "Inference Time Comparison", "Time (seconds)", output_dir / "inference_time_comparison.png", ascending=True, dpi=dpi)

    # 9. Actual vs Predicted Demand (Best Model)
    if best_preds_df is not None and not best_preds_df.empty:
        plot_actual_vs_predicted_best(best_preds_df, output_dir / "actual_vs_predicted_best_model.png", best_model_name=best_model_name, dpi=dpi)
        # Also output best_model_actual_vs_predicted.png for alternate naming compatibility
        plot_actual_vs_predicted_best(best_preds_df, output_dir / "best_model_actual_vs_predicted.png", best_model_name=best_model_name, dpi=dpi)

    # 10. Probabilistic Forecast (LightGBM Quantile)
    if quantile_preds_df is not None and not quantile_preds_df.empty:
        plot_probabilistic_forecast(quantile_preds_df, output_dir / "lightgbm_quantile_probabilistic_forecast.png", dpi=dpi)

    logger.info(f"All 10 publication-quality plots successfully saved to {output_dir}")
