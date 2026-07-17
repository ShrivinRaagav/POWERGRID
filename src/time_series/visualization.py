import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from src.config.settings import TS_PLOTS_DIR, TS_PLOTS_DPI, TS_DATE_COL, TS_TARGET_COL, TS_GROUP_BY
from src.utils.helpers import setup_logger

logger = setup_logger("visualization_engine")

def plot_group_decompositions(
    group_key: Tuple[Any, ...],
    df_group: pd.DataFrame,
    dwt_coeffs: Dict[str, np.ndarray],
    emd_results: Dict[str, np.ndarray],
    date_col: str,
    target_col: str,
    output_dir: Path,
    dpi: int = 150
):
    """
    Generates three separate plots for a representative time series group:
    1. Classical STL-like Decomposition (Original, Trend, Seasonal, Residual)
    2. DWT Coefficients (cA, cD_1, ..., cD_n)
    3. EMD IMFs & Residual
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    group_name = "_".join(str(k) for k in group_key)
    if not group_name or group_name == "global":
        group_name = "global"
        
    dates = pd.to_datetime(df_group[date_col]).values
    signal = df_group[target_col].values
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Classical Decomposition Plot
    if "Classical_Trend" in df_group.columns:
        trend = df_group["Classical_Trend"].values
        seasonal = df_group["Classical_Seasonal"].values
        resid = df_group["Classical_Residual"].values
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Representative Classical Seasonal Decomposition ({group_name})", fontsize=12, fontweight='bold')
        
        axes[0].plot(dates, signal, color='#2c3e50', linewidth=1.5, label="Original")
        axes[0].set_ylabel("Original")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        axes[1].plot(dates, trend, color='#e74c3c', linewidth=1.5, label="Trend")
        axes[1].set_ylabel("Trend")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, linestyle="--", alpha=0.5)
        
        axes[2].plot(dates, seasonal, color='#2ecc71', linewidth=1.5, label="Seasonal")
        axes[2].set_ylabel("Seasonal")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, linestyle="--", alpha=0.5)
        
        axes[3].scatter(dates, resid, color='#95a5a6', s=10, alpha=0.8, label="Residual")
        axes[3].plot(dates, resid, color='#95a5a6', linewidth=0.5, alpha=0.5)
        axes[3].set_ylabel("Residual")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, linestyle="--", alpha=0.5)
        
        plt.xlabel("Date")
        plt.tight_layout()
        
        plot_path = output_dir / f"{group_name}_classical_decomposition.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved representative classical plot to {plot_path}")
        
    # 2. DWT Coefficients Plot
    if dwt_coeffs:
        cA = dwt_coeffs["cA"]
        cD_keys = sorted([k for k in dwt_coeffs.keys() if k.startswith("cD_")], key=lambda x: int(x.split("_")[1]))
        
        num_dwt_plots = 1 + len(cD_keys)
        fig, axes = plt.subplots(num_dwt_plots, 1, figsize=(10, 1.8 * num_dwt_plots))
        fig.suptitle(f"Representative DWT Coefficients ({group_name})", fontsize=12, fontweight='bold')
        
        if num_dwt_plots == 1:
            axes = [axes]
            
        axes[0].plot(cA, color='#8e44ad', linewidth=1.5, label="Approximation (cA)")
        axes[0].set_ylabel("cA")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        for idx, k in enumerate(cD_keys):
            axes[idx + 1].plot(dwt_coeffs[k], color='#d35400', linewidth=1.5, label=f"Detail ({k})")
            axes[idx + 1].set_ylabel(k)
            axes[idx + 1].legend(loc="upper right")
            axes[idx + 1].grid(True, linestyle="--", alpha=0.5)
            
        plt.xlabel("Coefficient Index (Downsampled)")
        plt.tight_layout()
        
        plot_path = output_dir / f"{group_name}_dwt_decomposition.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved representative DWT plot to {plot_path}")
        
    # 3. EMD IMFs Plot
    if emd_results:
        imfs = emd_results["imfs"]
        residual = emd_results["residual"]
        num_imfs = imfs.shape[0]
        
        num_emd_plots = 1 + num_imfs + 1
        fig, axes = plt.subplots(num_emd_plots, 1, figsize=(10, 1.6 * num_emd_plots), sharex=True)
        fig.suptitle(f"Representative EMD IMFs & Residual ({group_name})", fontsize=12, fontweight='bold')
        
        axes[0].plot(dates, signal, color='#2c3e50', linewidth=1.5, label="Original Signal")
        axes[0].set_ylabel("Signal")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, linestyle="--", alpha=0.5)
        
        for idx in range(num_imfs):
            axes[idx + 1].plot(dates, imfs[idx], color='#3498db', linewidth=1.5, label=f"IMF {idx+1}")
            axes[idx + 1].set_ylabel(f"IMF {idx+1}")
            axes[idx + 1].legend(loc="upper right")
            axes[idx + 1].grid(True, linestyle="--", alpha=0.5)
            
        axes[-1].plot(dates, residual, color='#f1c40f', linewidth=1.5, label="Residual Trend")
        axes[-1].set_ylabel("Residual")
        axes[-1].legend(loc="upper right")
        axes[-1].grid(True, linestyle="--", alpha=0.5)
        
        plt.xlabel("Date")
        plt.tight_layout()
        
        plot_path = output_dir / f"{group_name}_emd_decomposition.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved representative EMD plot to {plot_path}")

def plot_correlation_heatmap(df_selected: pd.DataFrame, save_path: Path, dpi: int = 150):
    """Generates a correlation heatmap of the final selected forecasting features."""
    num_cols = df_selected.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in num_cols if c not in ["Date", "Project_ID", "Region", "Material_Type", "Quantity_Required"]]
    
    if not feature_cols:
        logger.warning("No feature columns available to plot correlation heatmap.")
        return
        
    corr = df_selected[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax)
    
    # Tick labels
    ticks = np.arange(len(feature_cols))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(feature_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(feature_cols, fontsize=8)
    
    ax.set_title("Selected Features Correlation Heatmap", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature correlation heatmap to {save_path}")

def plot_feature_importances(importances: Dict[str, float], save_path: Path, dpi: int = 150):
    """Plots a horizontal bar chart of the top selected features by importance score."""
    if not importances:
        logger.warning("No importances available to plot feature importances.")
        return
        
    # Sort and take top 15
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    names = [x[0] for x in sorted_imp]
    vals = [x[1] for x in sorted_imp]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(names))
    
    ax.barh(y_pos, vals, align="center", color="#3498db", edgecolor="#2980b9")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()  # Top-down orientation
    ax.set_xlabel("Importance Score")
    ax.set_title("Top 15 Most Informative Features", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved feature importances plot to {save_path}")

def generate_important_plots(
    df_enriched: pd.DataFrame,
    df_selected: pd.DataFrame,
    feature_selector: Any,
    dwt_transformer: Any,
    emd_processor: Any,
    group_cols: Optional[List[str]] = TS_GROUP_BY,
    date_col: str = TS_DATE_COL,
    target_col: str = TS_TARGET_COL,
    output_dir: Path = TS_PLOTS_DIR,
    dpi: int = TS_PLOTS_DPI
):
    """
    Orchestrates the generation of publication-quality plots:
    1. Selects the representative group (group with highest target variance).
    2. Plots its classical, DWT, and EMD decomposition signals.
    3. Plots correlation heatmap of final selected feature set.
    4. Plots feature importances for top selected features.
    """
    logger.info(f"Generating optimized visualizations in {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Identify representative group based on variance of Quantity_Required
    representative_key = ("global",)
    df_sorted = df_enriched.sort_values(by=[date_col]).copy()
    
    if group_cols:
        grouped = df_sorted.groupby(group_cols)
        best_var = -1.0
        for name, group in grouped:
            var = group[target_col].var()
            if var > best_var:
                best_var = var
                representative_key = name if isinstance(name, tuple) else (name,)
                
    # Plot decompositions for the representative group
    logger.info(f"Selected representative group {representative_key} for visualization.")
    
    df_rep = df_sorted
    if group_cols:
        # Filter rep group
        rep_cond = True
        if len(group_cols) == 1:
            rep_cond = df_sorted[group_cols[0]] == representative_key[0]
        else:
            for idx, col in enumerate(group_cols):
                rep_cond = rep_cond & (df_sorted[col] == representative_key[idx])
        df_rep = df_sorted[rep_cond]
        
    dwt_c = dwt_transformer.decomposition_results.get(representative_key, {})
    emd_r = emd_processor.decomposition_results.get(representative_key, {})
    
    plot_group_decompositions(
        group_key=representative_key,
        df_group=df_rep,
        dwt_coeffs=dwt_c,
        emd_results=emd_r,
        date_col=date_col,
        target_col=target_col,
        output_dir=output_dir,
        dpi=dpi
    )
    
    # 2. Plot correlation heatmap of selected features
    heatmap_path = output_dir / "selected_features_correlation_heatmap.png"
    plot_correlation_heatmap(df_selected, heatmap_path, dpi=dpi)
    
    # 3. Plot feature importances
    # Extract calculated importances from feature_selector metadata
    importances = {}
    for feat, meta in feature_selector.feature_metadata.items():
        importance = meta["Importance"]
        if not np.isnan(importance):
            importances[feat] = importance
            
    importance_path = output_dir / "feature_importances.png"
    plot_feature_importances(importances, importance_path, dpi=dpi)
    
    logger.info("Decomposition and feature selection plots successfully generated.")
import typing
Any = typing.Any
