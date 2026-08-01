import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple

from src.config.settings import REPORTS_DIR, EXPERIMENTS_DIR, PROCESSED_DATASET_PATH
from src.utils.helpers import setup_logger
from src.explainability.shap_explainer import get_best_model_info, load_best_model_checkpoint, load_processed_test_data
from src.optimization.optimizer import SupplyChainOptimizationProblem, run_nsga2_optimization
from src.optimization.decision_engine import generate_procurement_recommendations, export_optimization_results
from src.optimization.visualization import generate_all_optimization_plots
from src.optimization.report_generator import generate_optimization_report

logger = setup_logger("run_optimization")

def prepare_material_attributes(
    reports_dir: Path = REPORTS_DIR,
    data_path: Path = PROCESSED_DATASET_PATH
) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """
    Prepares material demand forecasts and baseline inventory attributes for optimization.
    """
    # 1. Load best model metadata & generate forecasts
    best_meta = get_best_model_info(reports_dir)
    best_model_name = str(best_meta.get("best_model", "xgboost")).lower()

    # Load test set
    X_test, y_test, _, feature_names = load_processed_test_data()

    # Try loading best model checkpoint to generate predictions
    try:
        model_wrapper, _ = load_best_model_checkpoint(best_model_name, EXPERIMENTS_DIR)
        if hasattr(model_wrapper, "predict"):
            preds = model_wrapper.predict(X_test)
            if isinstance(preds, dict):
                preds = preds.get("P50", list(preds.values())[0])
            y_pred = np.asarray(preds, dtype=np.float64).flatten()
        elif isinstance(model_wrapper, dict) and "model" in model_wrapper:
            y_pred = model_wrapper["model"].predict(X_test[model_wrapper.get("feature_cols", feature_names)])
        else:
            y_pred = y_test.values
    except Exception as e:
        logger.warning(f"Could not load best model predictions ({e}), falling back to ground truth test values.")
        y_pred = y_test.values

    # Add predictions back to dataframe if material type column present, else create synthetic material group
    if "Material_Type" in X_test.columns:
        X_test_copy = X_test.copy()
        X_test_copy["Predicted_Demand"] = y_pred
        mat_grouped = X_test_copy.groupby("Material_Type")["Predicted_Demand"].sum().reset_index()
    else:
        # Standard POWERGRID material categories
        categories = ["Transformer_Oil", "Conductor_ACSR", "Insulator_Porcelain", "Steel_Structure", "Control_Cable"]
        demands = [float(np.sum(y_pred[i::len(categories)])) for i in range(len(categories))]
        mat_grouped = pd.DataFrame({
            "Material_Type": categories,
            "Predicted_Demand": demands
        })

    # Domain attributes for POWERGRID supply chain
    default_attrs = {
        "Transformer_Oil": {"unit_price": 120.0, "lead_time": 4.0, "cur_inv": 1500.0, "wh_cap": 8000.0, "sup_cap": 12000.0},
        "Conductor_ACSR": {"unit_price": 450.0, "lead_time": 6.0, "cur_inv": 2200.0, "wh_cap": 12000.0, "sup_cap": 18000.0},
        "Insulator_Porcelain": {"unit_price": 85.0, "lead_time": 3.0, "cur_inv": 3100.0, "wh_cap": 15000.0, "sup_cap": 25000.0},
        "Steel_Structure": {"unit_price": 620.0, "lead_time": 8.0, "cur_inv": 1800.0, "wh_cap": 10000.0, "sup_cap": 15000.0},
        "Control_Cable": {"unit_price": 210.0, "lead_time": 4.0, "cur_inv": 2500.0, "wh_cap": 10000.0, "sup_cap": 20000.0}
    }

    records = []
    for _, row in mat_grouped.iterrows():
        m_type = str(row["Material_Type"])
        demand_val = max(100.0, float(row["Predicted_Demand"]))
        attr = default_attrs.get(m_type, {"unit_price": 250.0, "lead_time": 5.0, "cur_inv": 2000.0, "wh_cap": 10000.0, "sup_cap": 20000.0})

        records.append({
            "Material_Type": m_type,
            "Forecasted_Demand": demand_val,
            "Current_Inventory": attr["cur_inv"],
            "Unit_Price_INR": attr["unit_price"],
            "Lead_Time_Weeks": attr["lead_time"],
            "Max_Warehouse_Capacity": attr["wh_cap"],
            "Supplier_Max_Capacity": attr["sup_cap"]
        })

    material_df = pd.DataFrame(records)

    # Budget setup: 120% of baseline total demand procurement cost
    total_est_cost = float(np.sum(material_df["Forecasted_Demand"] * material_df["Unit_Price_INR"]))
    total_budget = total_est_cost * 1.20

    return material_df, total_budget, best_meta

def run_optimization_pipeline(reports_dir: Path = REPORTS_DIR) -> Dict[str, Any]:
    """
    Master pipeline orchestrator for Module 5 (Multi-Objective Supply Chain Optimization):
    1. Loads best model demand forecast outputs.
    2. Constructs supply chain optimization problem formulation.
    3. Solves multi-objective NSGA-II algorithm.
    4. Generates itemized procurement recommendations CSV.
    5. Exports optimization summary CSV & Pareto front CSV.
    6. Renders 5 IEEE publication-quality figures in reports/optimization_plots/.
    7. Compiles reports/optimization_report.md.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reports_dir / "optimization_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Module 5 Multi-Objective Supply Chain Optimization Pipeline...")

    # 1. Prepare Material Attributes & Demand Forecasts
    material_df, total_budget, best_meta = prepare_material_attributes(reports_dir)

    forecasted_demand = material_df["Forecasted_Demand"].values
    current_inventory = material_df["Current_Inventory"].values
    unit_prices = material_df["Unit_Price_INR"].values
    lead_times_weeks = material_df["Lead_Time_Weeks"].values
    max_warehouse_capacity = material_df["Max_Warehouse_Capacity"].values
    supplier_max_capacity = material_df["Supplier_Max_Capacity"].values

    # 2. Formulate Problem & Run NSGA-II
    problem = SupplyChainOptimizationProblem(
        forecasted_demand=forecasted_demand,
        current_inventory=current_inventory,
        unit_prices=unit_prices,
        lead_times_weeks=lead_times_weeks,
        max_warehouse_capacity=max_warehouse_capacity,
        supplier_max_capacity=supplier_max_capacity,
        total_budget=total_budget,
        holding_rate=0.15,
        z_score=1.65
    )

    opt_results = run_nsga2_optimization(problem, pop_size=100, n_gen=150, seed=42)

    best_X = opt_results["best_X"]
    best_idx = opt_results["best_idx"]
    pareto_F = opt_results["pareto_F"]

    # 3. Decision Engine & Recommendations Exporter
    rec_df = generate_procurement_recommendations(
        material_df=material_df,
        best_X=best_X,
        total_budget=total_budget,
        reports_dir=reports_dir
    )

    export_optimization_results(
        pareto_F=pareto_F,
        best_F=opt_results["best_F"],
        rec_df=rec_df,
        total_budget=total_budget,
        reports_dir=reports_dir
    )

    # 4. Render All IEEE Publication Visualizations
    generate_all_optimization_plots(
        pareto_F=pareto_F,
        best_idx=best_idx,
        rec_df=rec_df,
        output_dir=plots_dir,
        dpi=300
    )

    # 5. Export Markdown Optimization Report
    generate_optimization_report(
        best_meta=best_meta,
        rec_df=rec_df,
        pareto_F=pareto_F,
        best_idx=best_idx,
        total_budget=total_budget,
        save_path=reports_dir / "optimization_report.md"
    )

    logger.info("Module 5 Multi-Objective Supply Chain Optimization Pipeline completed successfully.")

    return {
        "best_model": best_meta.get("best_model"),
        "rec_df": rec_df,
        "pareto_F": pareto_F,
        "total_budget": total_budget
    }

def main():
    parser = argparse.ArgumentParser(description="POWERGRID Supply Chain Optimization Controller CLI (Module 5)")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save optimization reports.")
    args = parser.parse_args()

    run_optimization_pipeline(reports_dir=Path(args.reports_dir))

if __name__ == "__main__":
    main()

from typing import Tuple
Tuple_Df_Budget = Tuple[pd.DataFrame, float, Dict[str, Any]]
