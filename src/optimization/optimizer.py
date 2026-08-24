import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.utils.helpers import setup_logger
from src.optimization.objectives import evaluate_all_objectives
from src.optimization.constraints import evaluate_all_constraints

logger = setup_logger("supply_chain_optimizer")

class SupplyChainOptimizationProblem(ElementwiseProblem):
    """
    Pymoo multi-objective optimization problem formulation for POWERGRID supply chain:
    Variables:
      x[0 : N] -> Procurement Quantities Q_i
      x[N : 2N] -> Safety Stocks SS_i
    Objectives (5):
      f1 -> Total Procurement Cost
      f2 -> Total Holding Cost
      f3 -> Delivery Delay Risk
      f4 -> Service Level Deficit (1 - Service Level)
      f5 -> Stockout Risk Volume
    Constraints:
      g1: Warehouse Capacity Violations (N)
      g2: Budget Constraint (1)
      g3: Supplier Capacity Violations (N)
      g4: Safety Stock Violations (N)
    """
    def __init__(
        self,
        forecasted_demand: np.ndarray,
        current_inventory: np.ndarray,
        unit_prices: np.ndarray,
        lead_times_weeks: np.ndarray,
        max_warehouse_capacity: np.ndarray,
        supplier_max_capacity: np.ndarray,
        total_budget: float,
        holding_rate: float = 0.15,
        z_score: float = 1.65,
        demand_std: Optional[np.ndarray] = None
    ):
        self.n_items = len(forecasted_demand)
        self.forecasted_demand = np.asarray(forecasted_demand, dtype=np.float64)
        self.current_inventory = np.asarray(current_inventory, dtype=np.float64)
        self.unit_prices = np.asarray(unit_prices, dtype=np.float64)
        self.lead_times_weeks = np.asarray(lead_times_weeks, dtype=np.float64)
        self.max_warehouse_capacity = np.asarray(max_warehouse_capacity, dtype=np.float64)
        self.supplier_max_capacity = np.asarray(supplier_max_capacity, dtype=np.float64)
        self.total_budget = float(total_budget)
        self.holding_rate = float(holding_rate)

        if demand_std is None:
            self.demand_std = self.forecasted_demand * 0.15
        else:
            self.demand_std = np.asarray(demand_std, dtype=np.float64)

        # Minimum required safety stock: SS = z * sigma * sqrt(L)
        self.min_safety_stock = z_score * self.demand_std * np.sqrt(np.maximum(1.0, self.lead_times_weeks))

        # Variable Bounds:
        # Q_i in [0, 2.0 * Demand_i]
        # SS_i in [0, 1.5 * Demand_i]
        xl_q = np.zeros(self.n_items)
        xu_q = np.maximum(50.0, self.forecasted_demand * 2.5)

        xl_ss = np.zeros(self.n_items)
        xu_ss = np.maximum(20.0, self.forecasted_demand * 1.5)

        xl = np.concatenate([xl_q, xl_ss])
        xu = np.concatenate([xu_q, xu_ss])

        # Total constraints = N (warehouse) + 1 (budget) + N (supplier cap) + N (safety stock)
        n_constr = 3 * self.n_items + 1

        super().__init__(
            n_var=2 * self.n_items,
            n_obj=5,
            n_ieq_constr=n_constr,
            xl=xl,
            xu=xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        procurement_qty = x[:self.n_items]
        safety_stock = x[self.n_items:]

        # 1. Objectives
        objs = evaluate_all_objectives(
            procurement_qty=procurement_qty,
            current_inventory=self.current_inventory,
            forecasted_demand=self.forecasted_demand,
            unit_prices=self.unit_prices,
            lead_times_weeks=self.lead_times_weeks,
            supplier_capacity=self.supplier_max_capacity,
            holding_rate=self.holding_rate
        )

        out["F"] = [
            objs["Procurement_Cost"],
            objs["Holding_Cost"],
            objs["Delivery_Delay_Risk"],
            objs["Service_Level_Deficit"],
            objs["Stockout_Risk_Volume"]
        ]

        # 2. Constraints
        g_all, _ = evaluate_all_constraints(
            procurement_qty=procurement_qty,
            safety_stock=safety_stock,
            current_inventory=self.current_inventory,
            max_warehouse_capacity=self.max_warehouse_capacity,
            supplier_max_capacity=self.supplier_max_capacity,
            min_required_safety_stock=self.min_safety_stock,
            procurement_cost=objs["Procurement_Cost"],
            total_budget=self.total_budget
        )

        out["G"] = g_all

def select_compromise_solution(
    F: np.ndarray,
    X: np.ndarray,
    target_min_service_level: float = 95.0
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Selects the best compromise optimal solution from Pareto front using TOPSIS / Minimum Ideal Distance,
    prioritizing solutions satisfying POWERGRID's mission-critical target service level (>= 95%).
    """
    if len(F) == 0:
        raise ValueError("Pareto front F is empty.")

    service_levels = (1.0 - F[:, 3]) * 100.0
    feasible_indices = np.where(service_levels >= target_min_service_level)[0]
    
    if len(feasible_indices) == 0:
        # Fallback to top 15% highest service level solutions
        feasible_indices = np.argsort(service_levels)[-max(5, int(0.15 * len(F))):]

    sub_F = F[feasible_indices]
    f_min = np.min(sub_F, axis=0)
    f_max = np.max(sub_F, axis=0)
    denom = np.where((f_max - f_min) == 0, 1.0, f_max - f_min)
    F_norm = (sub_F - f_min) / denom

    # TOPSIS weights: cost efficiency while sustaining top-tier grid reliability
    weights = np.array([0.35, 0.20, 0.15, 0.20, 0.10])
    distances = np.linalg.norm(F_norm * weights, axis=1)
    best_sub_idx = int(np.argmin(distances))
    best_idx = int(feasible_indices[best_sub_idx])

    return best_idx, F[best_idx], X[best_idx]

def run_nsga2_optimization(
    problem: SupplyChainOptimizationProblem,
    pop_size: int = 100,
    n_gen: int = 150,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes NSGA-II algorithm on the supply chain optimization problem.
    """
    logger.info(f"Executing NSGA-II Optimization (pop_size={pop_size}, n_gen={n_gen})...")

    algorithm = NSGA2(
        pop_size=pop_size,
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", n_gen)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=False
    )

    logger.info(f"NSGA-II Optimization completed. Found {len(res.F)} Pareto optimal solutions.")

    best_idx, best_F, best_X = select_compromise_solution(res.F, res.X)

    return {
        "res": res,
        "pareto_F": res.F,
        "pareto_X": res.X,
        "best_idx": best_idx,
        "best_F": best_F,
        "best_X": best_X
    }
