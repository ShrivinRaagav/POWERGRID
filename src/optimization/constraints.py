import numpy as np
from typing import Dict, Any, Tuple

def evaluate_warehouse_capacity_constraint(
    procurement_qty: np.ndarray,
    current_inventory: np.ndarray,
    max_warehouse_capacity: np.ndarray
) -> np.ndarray:
    """
    g1(x): Total Inventory <= Warehouse Storage Limit
    g1(x) = (Current + Procurement) - Capacity <= 0
    """
    total_inv = current_inventory + procurement_qty
    return total_inv - max_warehouse_capacity

def evaluate_budget_constraint(
    procurement_cost: float,
    total_budget: float
) -> float:
    """
    g2(x): Total Procurement Cost <= Allocated Project Budget
    g2(x) = Procurement Cost - Budget <= 0
    """
    return procurement_cost - total_budget

def evaluate_supplier_capacity_constraint(
    procurement_qty: np.ndarray,
    supplier_max_capacity: np.ndarray
) -> np.ndarray:
    """
    g3(x): Procurement Quantity <= Supplier Max Production Capacity
    g3(x) = Procurement - Supplier Capacity <= 0
    """
    return procurement_qty - supplier_max_capacity

def evaluate_safety_stock_constraint(
    safety_stock: np.ndarray,
    min_required_safety_stock: np.ndarray
) -> np.ndarray:
    """
    g4(x): Safety Stock >= Required Minimum Safety Stock
    g4(x) = Min Required Safety Stock - Safety Stock <= 0
    """
    return min_required_safety_stock - safety_stock

def evaluate_all_constraints(
    procurement_qty: np.ndarray,
    safety_stock: np.ndarray,
    current_inventory: np.ndarray,
    max_warehouse_capacity: np.ndarray,
    supplier_max_capacity: np.ndarray,
    min_required_safety_stock: np.ndarray,
    procurement_cost: float,
    total_budget: float
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Evaluates all constraints for pymoo solver format.
    Returns:
      g_vec (np.ndarray): 1D array of constraint violations (satisfied if <= 0).
      info (Dict[str, Any]): Detailed violation breakdown for logging and reporting.
    """
    g_wh = evaluate_warehouse_capacity_constraint(procurement_qty, current_inventory, max_warehouse_capacity)
    g_bud = np.array([evaluate_budget_constraint(procurement_cost, total_budget)])
    g_sup = evaluate_supplier_capacity_constraint(procurement_qty, supplier_max_capacity)
    g_ss = evaluate_safety_stock_constraint(safety_stock, min_required_safety_stock)

    g_all = np.concatenate([g_wh, g_bud, g_sup, g_ss])

    info = {
        "max_warehouse_violation": float(np.max(np.maximum(0.0, g_wh))),
        "budget_violation": float(np.max(np.maximum(0.0, g_bud))),
        "max_supplier_violation": float(np.max(np.maximum(0.0, g_sup))),
        "max_safety_stock_violation": float(np.max(np.maximum(0.0, g_ss))),
        "total_constraint_violation": float(np.sum(np.maximum(0.0, g_all))),
        "is_feasible": bool(np.all(g_all <= 1e-5))
    }

    return g_all, info
