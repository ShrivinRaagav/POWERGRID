import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Any
from src.utils.helpers import setup_logger

logger = setup_logger("evaluation_visualization")

def plot_predictions_vs_actuals(
    y_true: Any,
    y_pred: Any,
    save_path: Union[str, Path],
    title: str = "Forecast Validation: Predictions vs Actuals",
    dpi: int = 150
):
    """
    Plots a time-series line comparison and a scatter comparison 
    of predicted vs actual values.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Line Plot (Sequence)
    axes[0].plot(y_t, label="Actual Demand", color="#2c3e50", linewidth=1.5)
    axes[0].plot(y_p, label="Predicted Demand", color="#e74c3c", linewidth=1.2, linestyle="--")
    axes[0].set_title("Timeline Comparison", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Quantity")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Scatter Plot
    axes[1].scatter(y_t, y_p, color="#3498db", alpha=0.6, s=15)
    # Perfect fit line
    lims = [
        np.min([axes[1].get_xlim(), axes[1].get_ylim()]),
        np.max([axes[1].get_xlim(), axes[1].get_ylim()])
    ]
    axes[1].plot(lims, lims, color="#2ecc71", linestyle="--", linewidth=1.5, label="Perfect Fit")
    axes[1].set_title("Actual vs Predicted Scatter", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved actual vs predicted validation plot to {save_p}")

def plot_residuals_distribution(
    y_true: Any,
    y_pred: Any,
    save_path: Union[str, Path],
    title: str = "Residuals Distribution Analysis",
    dpi: int = 150
):
    """
    Plots a histogram of forecast errors (residuals) and error values over time.
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    residuals = y_t - y_p
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Error values over time
    axes[0].plot(residuals, color="#8e44ad", linewidth=1.0)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[0].set_title("Forecast Errors Over Time", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Time Index")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Histogram of errors
    axes[1].hist(residuals, bins=25, density=True, color="#9b59b6", alpha=0.7, edgecolor="white")
    # Fit normal distribution line
    mean, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(np.min(residuals), np.max(residuals), 100)
    if std > 0:
        p = (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        axes[1].plot(x, p, color="#2c3e50", linewidth=1.5, label=f"Normal Fit\n(Mean={mean:.2f}, Std={std:.2f})")
    axes[1].set_title("Error Density Histogram", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Density")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved residuals distribution plot to {save_p}")

def plot_prediction_intervals(
    y_true: Any,
    y_pred: Any,
    lower_bound: Any,
    upper_bound: Any,
    save_path: Union[str, Path],
    title: str = "Forecast Validation with Prediction Intervals",
    dpi: int = 150
):
    """
    Plots a sequence comparison with shaded prediction intervals (confidence bands).
    """
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    y_l = np.asarray(lower_bound, dtype=np.float64).flatten()
    y_u = np.asarray(upper_bound, dtype=np.float64).flatten()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Timeline
    t = np.arange(len(y_t))
    
    # Shaded band
    ax.fill_between(t, y_l, y_u, color="#3498db", alpha=0.2, label="Prediction Interval Band")
    
    # Lines
    ax.plot(t, y_t, color="#2c3e50", linewidth=1.8, label="Actual Demand")
    ax.plot(t, y_p, color="#e74c3c", linewidth=1.5, linestyle="--", label="Forecast (Point)")
    ax.plot(t, y_l, color="#3498db", linewidth=0.8, alpha=0.5, label="Lower Bound")
    ax.plot(t, y_u, color="#3498db", linewidth=0.8, alpha=0.5, label="Upper Bound")
    
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Quantity")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved prediction intervals plot to {save_p}")
import typing
Any = typing.Any
