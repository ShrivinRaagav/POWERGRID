import numpy as np
from typing import Dict, Any, Union

def evaluate_procurement_cost(
    procurement_qty: np.ndarray,
    unit_prices: np.ndarray
) -> float:
    """
    Computes total material procurement cost.
    Procurement Cost = sum(Quantity * Unit Price)
    """
    return float(np.sum(procurement_qty * unit_prices))

def evaluate_holding_cost(
    procurement_qty: np.ndarray,
    current_inventory: np.ndarray,
    unit_prices: np.ndarray,
    holding_rate: float = 0.15
) -> float:
    """
    Computes total annual inventory holding cost.
    Average Inventory = Current Inventory + (Procurement Quantity / 2)
    Holding Cost = sum(Average Inventory * Unit Price * holding_rate)
    """
    avg_inventory = current_inventory + (procurement_qty / 2.0)
    return float(np.sum(avg_inventory * unit_prices * holding_rate))

def evaluate_delivery_delay_risk(
    procurement_qty: np.ndarray,
    lead_times_weeks: np.ndarray,
    supplier_capacity: np.ndarray
) -> float:
    """
    Evaluates delivery delay risk score.
    Over-allocating beyond 80% of supplier production capacity penalizes lead time exponentially.
    """
    capacity_utilization = procurement_qty / (supplier_capacity + 1e-6)
    delay_penalty = np.maximum(0.0, capacity_utilization - 0.8) ** 2
    delay_score = lead_times_weeks * (1.0 + 5.0 * delay_penalty)
    return float(np.mean(delay_score))

def evaluate_service_level_deficit(
    procurement_qty: np.ndarray,
    current_inventory: np.ndarray,
    forecasted_demand: np.ndarray
) -> float:
    """
    Computes Service Level Deficit (1.0 - Service Level).
    Service Level = min(Available Inventory, Forecasted Demand) / Forecasted Demand
    Deficit = 1.0 - Mean(Service Level)
    """
    available_inv = current_inventory + procurement_qty
    fill_rate = np.minimum(available_inv, forecasted_demand) / (forecasted_demand + 1e-6)
    service_level = np.clip(fill_rate, 0.0, 1.0)
    return float(1.0 - np.mean(service_level))

def evaluate_stockout_risk(
    procurement_qty: np.ndarray,
    current_inventory: np.ndarray,
    forecasted_demand: np.ndarray
) -> float:
    """
    Computes total stockout shortage risk volume in units.
    Shortage = max(0, Forecasted Demand - Available Inventory)
    """
    available_inv = current_inventory + procurement_qty
    shortage = np.maximum(0.0, forecasted_demand - available_inv)
    return float(np.sum(shortage))

def evaluate_all_objectives(
    procurement_qty: np.ndarray,
    current_inventory: np.ndarray,
    forecasted_demand: np.ndarray,
    unit_prices: np.ndarray,
    lead_times_weeks: np.ndarray,
    supplier_capacity: np.ndarray,
    holding_rate: float = 0.15
) -> Dict[str, float]:
    """
    Helper function calculating all 5 supply chain objectives for a given decision vector.
    """
    p_cost = evaluate_procurement_cost(procurement_qty, unit_prices)
    h_cost = evaluate_holding_cost(procurement_qty, current_inventory, unit_prices, holding_rate)
    delay_risk = evaluate_delivery_delay_risk(procurement_qty, lead_times_weeks, supplier_capacity)
    sl_deficit = evaluate_service_level_deficit(procurement_qty, current_inventory, forecasted_demand)
    stockout = evaluate_stockout_risk(procurement_qty, current_inventory, forecasted_demand)

    return {
        "Procurement_Cost": p_cost,
        "Holding_Cost": h_cost,
        "Delivery_Delay_Risk": delay_risk,
        "Service_Level_Deficit": sl_deficit,
        "Service_Level_Pct": (1.0 - sl_deficit) * 100.0,
        "Stockout_Risk_Volume": stockout
    }
