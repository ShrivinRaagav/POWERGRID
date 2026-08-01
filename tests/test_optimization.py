import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
from pathlib import Path

from src.optimization.objectives import (
    evaluate_procurement_cost, evaluate_holding_cost, evaluate_delivery_delay_risk,
    evaluate_service_level_deficit, evaluate_stockout_risk, evaluate_all_objectives
)
from src.optimization.constraints import (
    evaluate_warehouse_capacity_constraint, evaluate_budget_constraint,
    evaluate_supplier_capacity_constraint, evaluate_safety_stock_constraint, evaluate_all_constraints
)
from src.optimization.optimizer import SupplyChainOptimizationProblem, run_nsga2_optimization, select_compromise_solution
from src.optimization.decision_engine import generate_procurement_recommendations, export_optimization_results
from src.optimization.visualization import generate_all_optimization_plots
from src.optimization.report_generator import generate_optimization_report

class TestSupplyChainOptimization(unittest.TestCase):
    """Unit test suite for Module 5 Multi-Objective Supply Chain Optimization."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reports_dir = self.temp_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.procurement_qty = np.array([500.0, 1200.0, 800.0])
        self.current_inventory = np.array([200.0, 500.0, 300.0])
        self.forecasted_demand = np.array([600.0, 1500.0, 1000.0])
        self.unit_prices = np.array([100.0, 250.0, 150.0])
        self.lead_times_weeks = np.array([4.0, 6.0, 3.0])
        self.max_warehouse_capacity = np.array([2000.0, 3000.0, 2500.0])
        self.supplier_max_capacity = np.array([5000.0, 5000.0, 5000.0])
        self.safety_stock = np.array([100.0, 200.0, 150.0])
        self.min_safety_stock = np.array([50.0, 100.0, 80.0])
        self.total_budget = 1000000.0

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_objective_evaluations(self):
        p_cost = evaluate_procurement_cost(self.procurement_qty, self.unit_prices)
        self.assertEqual(p_cost, 500*100 + 1200*250 + 800*150)

        h_cost = evaluate_holding_cost(self.procurement_qty, self.current_inventory, self.unit_prices, holding_rate=0.15)
        self.assertGreater(h_cost, 0.0)

        delay_risk = evaluate_delivery_delay_risk(self.procurement_qty, self.lead_times_weeks, self.supplier_max_capacity)
        self.assertGreaterEqual(delay_risk, 0.0)

        sl_deficit = evaluate_service_level_deficit(self.procurement_qty, self.current_inventory, self.forecasted_demand)
        self.assertGreaterEqual(sl_deficit, 0.0)
        self.assertLessEqual(sl_deficit, 1.0)

        stockout = evaluate_stockout_risk(self.procurement_qty, self.current_inventory, self.forecasted_demand)
        self.assertGreaterEqual(stockout, 0.0)

        objs = evaluate_all_objectives(
            self.procurement_qty, self.current_inventory, self.forecasted_demand,
            self.unit_prices, self.lead_times_weeks, self.supplier_max_capacity
        )
        self.assertIn("Procurement_Cost", objs)
        self.assertIn("Service_Level_Pct", objs)

    def test_constraint_evaluations(self):
        g_all, info = evaluate_all_constraints(
            procurement_qty=self.procurement_qty,
            safety_stock=self.safety_stock,
            current_inventory=self.current_inventory,
            max_warehouse_capacity=self.max_warehouse_capacity,
            supplier_max_capacity=self.supplier_max_capacity,
            min_required_safety_stock=self.min_safety_stock,
            procurement_cost=500000.0,
            total_budget=1000000.0
        )
        self.assertTrue(info["is_feasible"])
        self.assertLessEqual(info["total_constraint_violation"], 1e-5)

    def test_nsga2_optimizer(self):
        problem = SupplyChainOptimizationProblem(
            forecasted_demand=self.forecasted_demand,
            current_inventory=self.current_inventory,
            unit_prices=self.unit_prices,
            lead_times_weeks=self.lead_times_weeks,
            max_warehouse_capacity=self.max_warehouse_capacity,
            supplier_max_capacity=self.supplier_max_capacity,
            total_budget=self.total_budget,
            holding_rate=0.15,
            z_score=1.65
        )

        res_dict = run_nsga2_optimization(problem, pop_size=20, n_gen=10, seed=42)

        self.assertIn("pareto_F", res_dict)
        self.assertIn("best_X", res_dict)
        self.assertEqual(len(res_dict["best_X"]), 6)  # 3 procurement_qty + 3 safety_stock

    def test_decision_engine_exports(self):
        mat_df = pd.DataFrame({
            "Material_Type": ["Mat_A", "Mat_B", "Mat_C"],
            "Forecasted_Demand": self.forecasted_demand,
            "Current_Inventory": self.current_inventory,
            "Unit_Price_INR": self.unit_prices,
            "Lead_Time_Weeks": self.lead_times_weeks,
            "Max_Warehouse_Capacity": self.max_warehouse_capacity,
            "Supplier_Max_Capacity": self.supplier_max_capacity
        })
        best_X = np.array([500.0, 1200.0, 800.0, 100.0, 200.0, 150.0])
        pareto_F = np.array([[500000.0, 50000.0, 2.5, 0.05, 0.0]])

        rec_df = generate_procurement_recommendations(mat_df, best_X, self.total_budget, self.reports_dir)
        self.assertTrue((self.reports_dir / "procurement_recommendations.csv").exists())

        export_optimization_results(pareto_F, pareto_F[0], rec_df, self.total_budget, self.reports_dir)
        self.assertTrue((self.reports_dir / "optimization_results.csv").exists())
        self.assertTrue((self.reports_dir / "pareto_front.csv").exists())

    def test_visualization_generation(self):
        rec_df = pd.DataFrame({
            "Material_Type": ["Mat_A", "Mat_B"],
            "Forecasted_Demand": [600.0, 1500.0],
            "Current_Inventory": [200.0, 500.0],
            "Recommended_Procurement_Qty": [500.0, 1200.0],
            "Safety_Stock_Qty": [100.0, 200.0],
            "Reorder_Point": [150.0, 350.0],
            "Expected_Inventory_Level": [100.0, 200.0],
            "Estimated_Procurement_Cost_INR": [50000.0, 300000.0],
            "Service_Level_Pct": [98.0, 95.0]
        })
        pareto_F = np.array([[350000.0, 20000.0, 1.5, 0.03, 0.0]])
        plots_dir = self.reports_dir / "optimization_plots"

        generate_all_optimization_plots(pareto_F, 0, rec_df, plots_dir, dpi=100)

        expected_plots = [
            "pareto_front.png",
            "inventory_comparison.png",
            "procurement_quantity.png",
            "cost_breakdown.png",
            "service_level_comparison.png"
        ]
        for plot_name in expected_plots:
            f_path = plots_dir / plot_name
            self.assertTrue(f_path.exists(), f"Missing plot artifact: {plot_name}")
            self.assertGreater(f_path.stat().st_size, 0, f"Empty plot artifact: {plot_name}")

    def test_report_generation(self):
        rec_df = pd.DataFrame({
            "Material_Type": ["Mat_A"],
            "Forecasted_Demand": [600.0],
            "Current_Inventory": [200.0],
            "Recommended_Procurement_Qty": [500.0],
            "Safety_Stock_Qty": [100.0],
            "Reorder_Point": [150.0],
            "Expected_Inventory_Level": [100.0],
            "Estimated_Procurement_Cost_INR": [50000.0],
            "Service_Level_Pct": [98.0]
        })
        pareto_F = np.array([[50000.0, 5000.0, 1.0, 0.02, 0.0]])
        best_meta = {"best_model": "xgboost"}
        save_p = self.reports_dir / "optimization_report.md"

        generate_optimization_report(best_meta, rec_df, pareto_F, 0, 100000.0, save_p)

        self.assertTrue(save_p.exists())
        content = save_p.read_text(encoding="utf-8")
        self.assertIn("Multi-Objective Supply Chain Optimization Report", content)

if __name__ == "__main__":
    unittest.main()
