import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from src.utils.helpers import setup_logger
from src.optimization.objectives import evaluate_all_objectives

logger = setup_logger("decision_engine")

def generate_procurement_recommendations(
    material_df: pd.DataFrame,
    best_X: np.ndarray,
    total_budget: float,
    reports_dir: Path
) -> pd.DataFrame:
    """
    Translates NSGA-II compromise decision vector into itemized procurement recommendations.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    n_items = len(material_df)

    procurement_qty = best_X[:n_items]
    safety_stock = best_X[n_items:]

    rec_df = material_df.copy()

    rec_df["Forecasted_Demand"] = rec_df["Forecasted_Demand"].astype(float)
    rec_df["Current_Inventory"] = rec_df["Current_Inventory"].astype(float)
    rec_df["Unit_Price_INR"] = rec_df["Unit_Price_INR"].astype(float)
    rec_df["Lead_Time_Weeks"] = rec_df["Lead_Time_Weeks"].astype(float)

    rec_df["Recommended_Procurement_Qty"] = np.round(procurement_qty, 2)
    rec_df["Safety_Stock_Qty"] = np.round(safety_stock, 2)

    # Reorder Point = Weekly_Demand * Lead_Time + Safety_Stock (planning horizon is 4 weeks)
    weekly_demand = rec_df["Forecasted_Demand"] / 4.0
    rec_df["Reorder_Point"] = np.round((weekly_demand * rec_df["Lead_Time_Weeks"]) + rec_df["Safety_Stock_Qty"], 2)

    # Expected Inventory Level at end of cycle = Current Inventory + Procurement Qty - Forecasted Demand
    rec_df["Expected_Inventory_Level"] = np.round(
        np.maximum(0.0, rec_df["Current_Inventory"] + rec_df["Recommended_Procurement_Qty"] - rec_df["Forecasted_Demand"]), 2
    )

    # Item Procurement Cost
    rec_df["Estimated_Procurement_Cost_INR"] = np.round(
        rec_df["Recommended_Procurement_Qty"] * rec_df["Unit_Price_INR"], 2
    )

    total_procurement_cost = float(rec_df["Estimated_Procurement_Cost_INR"].sum())
    rec_df["Budget_Utilization_Pct"] = np.round((rec_df["Estimated_Procurement_Cost_INR"] / total_budget) * 100.0, 4)

    # Service Level Per Item
    available_inv = rec_df["Current_Inventory"] + rec_df["Recommended_Procurement_Qty"]
    fill_rate = np.minimum(available_inv, rec_df["Forecasted_Demand"]) / (rec_df["Forecasted_Demand"] + 1e-6)
    rec_df["Service_Level_Pct"] = np.round(np.clip(fill_rate, 0.0, 1.0) * 100.0, 2)

    # Export procurement_recommendations.csv
    csv_path = reports_dir / "procurement_recommendations.csv"
    rec_df.to_csv(csv_path, index=False)
    logger.info(f"Exported procurement recommendations to {csv_path}")

    return rec_df

def export_optimization_results(
    pareto_F: np.ndarray,
    best_F: np.ndarray,
    rec_df: pd.DataFrame,
    total_budget: float,
    reports_dir: Path
):
    """
    Exports reports/optimization_results.csv and reports/pareto_front.csv.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pareto Front CSV
    pareto_df = pd.DataFrame(
        pareto_F,
        columns=[
            "Procurement_Cost_INR",
            "Holding_Cost_INR",
            "Delivery_Delay_Risk",
            "Service_Level_Deficit",
            "Stockout_Risk_Volume"
        ]
    )
    pareto_df["Service_Level_Pct"] = (1.0 - pareto_df["Service_Level_Deficit"]) * 100.0
    pareto_csv = reports_dir / "pareto_front.csv"
    pareto_df.to_csv(pareto_csv, index=False)
    logger.info(f"Exported Pareto front solutions to {pareto_csv}")

    # 2. Optimization Summary Results CSV
    total_procurement_cost = float(rec_df["Estimated_Procurement_Cost_INR"].sum())
    total_holding_cost = float(best_F[1])
    total_delay_risk = float(best_F[2])
    mean_service_level = float(rec_df["Service_Level_Pct"].mean())
    budget_utilization = float((total_procurement_cost / total_budget) * 100.0)

    summary_df = pd.DataFrame([{
        "Metric": "Total Procurement Cost (INR)", "Value": total_procurement_cost
    }, {
        "Metric": "Total Inventory Holding Cost (INR)", "Value": total_holding_cost
    }, {
        "Metric": "Total Supply Chain Cost (INR)", "Value": total_procurement_cost + total_holding_cost
    }, {
        "Metric": "Delivery Delay Risk Index", "Value": total_delay_risk
    }, {
        "Metric": "Mean Service Level (%)", "Value": mean_service_level
    }, {
        "Metric": "Total Allocated Budget (INR)", "Value": total_budget
    }, {
        "Metric": "Budget Utilization (%)", "Value": budget_utilization
    }, {
        "Metric": "Pareto Optimal Solutions Count", "Value": len(pareto_F)
    }])

    res_csv = reports_dir / "optimization_results.csv"
    summary_df.to_csv(res_csv, index=False)
    logger.info(f"Exported optimization summary results to {res_csv}")
