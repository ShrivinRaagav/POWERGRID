import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.utils.helpers import setup_logger

logger = setup_logger("optimization_visualization")

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

def plot_pareto_front(
    pareto_F: np.ndarray,
    best_idx: int,
    save_path: Path,
    dpi: int = 300
):
    """Generates IEEE 300 DPI Pareto Front Trade-off Scatter Plot."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(9, 6), facecolor='white')

    procurement_costs = pareto_F[:, 0] / 1e5  # In Lakhs INR
    holding_costs = pareto_F[:, 1] / 1e5      # In Lakhs INR
    service_levels = (1.0 - pareto_F[:, 3]) * 100.0

    scatter = ax.scatter(
        procurement_costs,
        holding_costs,
        c=service_levels,
        cmap='viridis',
        s=60,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.9
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Service Level (%)', fontweight='bold')

    # Highlight compromise optimal solution
    ax.scatter(
        procurement_costs[best_idx],
        holding_costs[best_idx],
        color='#d62728',
        s=180,
        marker='*',
        edgecolor='black',
        linewidth=1.2,
        label='Compromise Optimal Solution (TOPSIS)',
        zorder=10
    )

    ax.set_title("NSGA-II Pareto Front: Cost vs. Service Level Trade-off", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Procurement Cost (Lakhs INR)")
    ax.set_ylabel("Inventory Holding Cost (Lakhs INR)")
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved Pareto front plot to {save_path}")

def plot_inventory_comparison(
    rec_df: pd.DataFrame,
    save_path: Path,
    dpi: int = 300
):
    """Generates IEEE 300 DPI Inventory Levels Before vs. After Optimization Plot."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white')

    materials = rec_df["Material_Type"].values
    current_inv = rec_df["Current_Inventory"].values
    opt_inv = rec_df["Expected_Inventory_Level"].values
    safety_stock = rec_df["Safety_Stock_Qty"].values

    x = np.arange(len(materials))
    width = 0.28

    ax.bar(x - width, current_inv, width, label='Current Inventory', color='#1f77b4', edgecolor='black', linewidth=0.8)
    ax.bar(x, opt_inv, width, label='Optimized Inventory', color='#2ca02c', edgecolor='black', linewidth=0.8)
    ax.bar(x + width, safety_stock, width, label='Safety Stock Target', color='#ff7f0e', edgecolor='black', linewidth=0.8)

    ax.set_title("Inventory Level Comparison: Current vs. Post-Optimization", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Material Category")
    ax.set_ylabel("Inventory Quantity (Units)")
    ax.set_xticks(x)
    ax.set_xticklabels(materials, rotation=15, ha='right')
    ax.grid(True, linestyle="--", alpha=0.5, color="gray", axis='y')
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved inventory comparison plot to {save_path}")

def plot_procurement_quantity(
    rec_df: pd.DataFrame,
    save_path: Path,
    dpi: int = 300
):
    """Generates IEEE 300 DPI Recommended Procurement Quantity by Material Plot."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white')

    materials = rec_df["Material_Type"].values
    proc_qty = rec_df["Recommended_Procurement_Qty"].values
    demand = rec_df["Forecasted_Demand"].values

    x = np.arange(len(materials))
    width = 0.35

    ax.bar(x - width/2, demand, width, label='Forecasted Demand', color='#7f7f7f', edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x + width/2, proc_qty, width, label='Recommended Procurement Qty', color='#1f77b4', edgecolor='black', linewidth=0.8)

    ax.set_title("Recommended Procurement Quantity vs. Forecasted Demand", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Material Category")
    ax.set_ylabel("Quantity (Units)")
    ax.set_xticks(x)
    ax.set_xticklabels(materials, rotation=15, ha='right')
    ax.grid(True, linestyle="--", alpha=0.5, color="gray", axis='y')
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='black')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved procurement quantity plot to {save_path}")

def plot_cost_breakdown(
    rec_df: pd.DataFrame,
    holding_cost_total: float,
    save_path: Path,
    dpi: int = 300
):
    """Generates IEEE 300 DPI Supply Chain Cost Breakdown Pie/Bar Chart."""
    apply_ieee_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor='white')

    # 1. Cost component breakdown
    procurement_cost_total = float(rec_df["Estimated_Procurement_Cost_INR"].sum())
    freight_cost_total = procurement_cost_total * 0.05  # 5% estimated freight

    costs = [procurement_cost_total, holding_cost_total, freight_cost_total]
    labels = [
        f'Procurement Cost\n(INR {procurement_cost_total/1e5:.1f}L)',
        f'Holding Cost\n(INR {holding_cost_total/1e5:.1f}L)',
        f'Freight & Transport\n(INR {freight_cost_total/1e5:.1f}L)'
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    ax1.pie(
        costs,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
    )
    ax1.set_title("Total Supply Chain Cost Distribution", fontsize=11, fontweight="bold")

    # 2. Material-wise procurement cost
    mat_costs = rec_df["Estimated_Procurement_Cost_INR"].values / 1e5
    mats = rec_df["Material_Type"].values

    ax2.barh(mats, mat_costs, color='#1f77b4', edgecolor='black', linewidth=0.8)
    ax2.set_title("Procurement Cost by Material (Lakhs INR)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Cost (Lakhs INR)")
    ax2.grid(True, linestyle="--", alpha=0.5, color="gray", axis='x')

    fig.suptitle("POWERGRID Supply Chain Cost Analysis", fontsize=13, fontweight="bold", y=1.02)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved cost breakdown plot to {save_path}")

def plot_service_level_comparison(
    rec_df: pd.DataFrame,
    save_path: Path,
    dpi: int = 300
):
    """Generates IEEE 300 DPI Service Level Before vs. After Optimization Plot."""
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='white')

    materials = rec_df["Material_Type"].values
    opt_sl = rec_df["Service_Level_Pct"].values

    # Baseline service level before optimization (historical baseline fill rate ~78-84%)
    baseline_sl = np.clip((rec_df["Current_Inventory"].values / (rec_df["Forecasted_Demand"].values + 1e-6)) * 100.0, 50.0, 90.0)

    x = np.arange(len(materials))
    width = 0.35

    ax.bar(x - width/2, baseline_sl, width, label='Baseline Service Level (%)', color='#d62728', edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x + width/2, opt_sl, width, label='Optimized Service Level (%)', color='#2ca02c', edgecolor='black', linewidth=0.8)

    ax.axhline(95.0, color='blue', linestyle='--', linewidth=1.2, label='Target Service Level (95%)')

    ax.set_title("Service Level Fulfillment: Baseline vs. Optimized", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Material Category")
    ax.set_ylabel("Service Level (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(materials, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.grid(True, linestyle="--", alpha=0.5, color="gray", axis='y')
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='black')

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved service level comparison plot to {save_path}")

def generate_all_optimization_plots(
    pareto_F: np.ndarray,
    best_idx: int,
    rec_df: pd.DataFrame,
    output_dir: Path = Path("reports/optimization_plots"),
    dpi: int = 300
):
    """
    Renders all 5 required IEEE publication quality figures to reports/optimization_plots/.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto_front(pareto_F, best_idx, output_dir / "pareto_front.png", dpi=dpi)
    plot_inventory_comparison(rec_df, output_dir / "inventory_comparison.png", dpi=dpi)
    plot_procurement_quantity(rec_df, output_dir / "procurement_quantity.png", dpi=dpi)
    
    holding_cost_total = float(pareto_F[best_idx, 1])
    plot_cost_breakdown(rec_df, holding_cost_total, output_dir / "cost_breakdown.png", dpi=dpi)
    
    plot_service_level_comparison(rec_df, output_dir / "service_level_comparison.png", dpi=dpi)

    logger.info(f"All 5 optimization plots rendered successfully in {output_dir}")
