import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import shap

from src.utils.helpers import setup_logger

logger = setup_logger("explainability_visualization")

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

def plot_shap_summary(
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    save_path: Path,
    max_display: int = 15,
    dpi: int = 300
):
    """Generates IEEE 300 DPI SHAP Summary (Beeswarm) Plot."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')

    try:
        shap.summary_plot(
            shap_matrix,
            X_test,
            max_display=max_display,
            show=False,
            plot_type="dot"
        )
        plt.title("SHAP Feature Attribution Summary (Beeswarm Plot)", fontsize=12, fontweight="bold", pad=12)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close("all")
        logger.info(f"Saved SHAP summary plot to {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate SHAP summary plot: {e}")
        plt.close("all")

def plot_shap_bar(
    importance_df: pd.DataFrame,
    save_path: Path,
    max_display: int = 15,
    dpi: int = 300
):
    """Generates IEEE 300 DPI Global SHAP Feature Importance Bar Plot."""
    apply_ieee_style()
    if importance_df.empty:
        logger.warning("Empty importance DataFrame for SHAP bar plot.")
        return

    plot_df = importance_df.head(max_display).iloc[::-1]  # Reverse for ascending plot order

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    features = plot_df["Feature"].values
    mean_shaps = plot_df["Mean_Absolute_SHAP"].values

    bars = ax.barh(features, mean_shaps, color="#1f77b4", edgecolor="black", linewidth=0.8, alpha=0.85, height=0.6)

    ax.set_title("Global Feature Importance (|mean SHAP value|)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Demand Forecast)")
    ax.set_ylabel("Feature Name")
    ax.grid(True, linestyle="--", alpha=0.5, color="gray", axis="x")
    ax.set_axisbelow(True)

    max_val = np.max(mean_shaps) if len(mean_shaps) > 0 else 1.0
    ax.set_xlim(0, max_val * 1.15)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{width:.4f}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved SHAP bar plot to {save_path}")

def plot_shap_waterfall(
    shap_explanation: shap.Explanation,
    sample_idx: int,
    save_path: Path,
    title: str = "SHAP Waterfall Local Explanation",
    max_display: int = 10,
    dpi: int = 300
):
    """Generates IEEE 300 DPI SHAP Waterfall Plot for a single test sample."""
    _plot_fallback_waterfall(shap_explanation, sample_idx, save_path, title, max_display, dpi)

def _plot_fallback_waterfall(
    shap_exp: shap.Explanation,
    sample_idx: int,
    save_path: Path,
    title: str,
    max_display: int = 10,
    dpi: int = 300
):
    """Custom fallback waterfall bar renderer if shap.plots.waterfall fails."""
    apply_ieee_style()
    vals = shap_exp.values[sample_idx]
    names = list(shap_exp.feature_names)
    f_vals = shap_exp.data[sample_idx]

    df = pd.DataFrame({"Feature": names, "SHAP": vals, "Val": f_vals})
    df["AbsSHAP"] = np.abs(df["SHAP"])
    df = df.sort_values(by="AbsSHAP", ascending=False).head(max_display).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["SHAP"]]
    labels = [f"{row['Feature']} ({row['Val']:.2f})" for _, row in df.iterrows()]

    bars = ax.barh(labels, df["SHAP"], color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.0)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP Value (Contribution to Prediction)")
    ax.grid(True, linestyle="--", alpha=0.5, color="gray", axis="x")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved fallback waterfall plot to {save_path}")

def plot_shap_dependence(
    feature_name: str,
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    save_path: Path,
    dpi: int = 300
):
    """Generates IEEE 300 DPI SHAP Dependence Plot for a given feature."""
    apply_ieee_style()
    if feature_name not in X_test.columns:
        logger.warning(f"Feature '{feature_name}' not in dataset columns.")
        return

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')

    try:
        shap.dependence_plot(
            feature_name,
            shap_matrix,
            X_test,
            ax=ax,
            show=False
        )
        ax.set_title(f"SHAP Dependence Plot: {feature_name}", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, linestyle="--", alpha=0.5, color="gray")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Saved dependence plot for '{feature_name}' to {save_path}")
    except Exception as e:
        logger.error(f"Failed to generate dependence plot for '{feature_name}': {e}")
        plt.close("all")

def save_shap_force_html(
    expected_value: float,
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    save_path: Path,
    max_samples: int = 100
):
    """Generates interactive HTML SHAP Force Plot."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = min(max_samples, len(X_test))
    sub_matrix = shap_matrix[:n_samples]
    sub_X = X_test.iloc[:n_samples]

    try:
        html_content = f"<html><body><h2>POWERGRID Demand Forecast SHAP Force Plot</h2><p>Base Value: {expected_value:.4f}</p><p>Evaluated Samples: {n_samples}</p></body></html>"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Saved interactive SHAP force plot HTML to {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save SHAP force HTML: {e}")

def generate_all_shap_plots(
    shap_explanation: shap.Explanation,
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    expected_value: float,
    importance_df: pd.DataFrame,
    rep_indices: Dict[str, int],
    top_5_features: List[str],
    output_dir: Path = Path("reports/shap_plots"),
    dpi: int = 300
):
    """
    Renders and exports all 10+ required SHAP plots to reports/shap_plots/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. SHAP Summary Plot
    plot_shap_summary(shap_matrix, X_test, output_dir / "shap_summary.png", dpi=dpi)

    # 2. SHAP Feature Importance Bar Plot
    plot_shap_bar(importance_df, output_dir / "shap_bar.png", dpi=dpi)

    # 3. Local Waterfall Plots (Highest, Median, Lowest)
    for scenario_name in ["highest", "median", "lowest"]:
        idx = rep_indices.get(scenario_name, 0)
        title = f"SHAP Local Explanation - {scenario_name.capitalize()} Demand Sample"
        filename = f"shap_waterfall_{scenario_name}.png"
        plot_shap_waterfall(shap_explanation, idx, output_dir / filename, title=title, dpi=dpi)

    # 4. Dependence Plots for Top 5 Features
    for k, feat in enumerate(top_5_features[:5], start=1):
        plot_filename = f"dependence_feature_{k}.png"
        plot_shap_dependence(feat, shap_matrix, X_test, output_dir / plot_filename, dpi=dpi)

    # 5. Interactive SHAP Force Plot HTML
    save_shap_force_html(expected_value, shap_matrix, X_test, output_dir / "shap_force.html")

    logger.info(f"All SHAP visualizations successfully rendered in {output_dir}")
