import numpy as np
import pandas as pd
from typing import Dict, Union, Optional, List, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def to_numpy_arrays(
    y_true: Union[np.ndarray, pd.Series, List],
    y_pred: Union[np.ndarray, pd.Series, List]
) -> tuple[np.ndarray, np.ndarray]:
    """Ensures y_true and y_pred are 1D numpy float64 arrays of matching lengths."""
    y_t = np.asarray(y_true, dtype=np.float64).flatten()
    y_p = np.asarray(y_pred, dtype=np.float64).flatten()
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true length {len(y_t)} != y_pred length {len(y_p)}")
    return y_t, y_p

def calculate_mae(y_true: Any, y_pred: Any) -> float:
    """Calculates Mean Absolute Error."""
    y_t, y_p = to_numpy_arrays(y_true, y_pred)
    return float(mean_absolute_error(y_t, y_p))

def calculate_rmse(y_true: Any, y_pred: Any) -> float:
    """Calculates Root Mean Squared Error."""
    y_t, y_p = to_numpy_arrays(y_true, y_pred)
    return float(np.sqrt(mean_squared_error(y_t, y_p)))

def calculate_mape(y_true: Any, y_pred: Any, epsilon: float = 1e-5) -> float:
    """
    Calculates Mean Absolute Percentage Error.
    Uses epsilon in denominator to avoid division by zero.
    """
    y_t, y_p = to_numpy_arrays(y_true, y_pred)
    # Avoid zero values in true target using epsilon
    denom = np.where(np.abs(y_t) < epsilon, epsilon, y_t)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)

def calculate_r2(y_true: Any, y_pred: Any) -> float:
    """Calculates R² Coefficient of Determination."""
    y_t, y_p = to_numpy_arrays(y_true, y_pred)
    return float(r2_score(y_t, y_p))

def calculate_smape(y_true: Any, y_pred: Any, epsilon: float = 1e-5) -> float:
    """
    Calculates Symmetric Mean Absolute Percentage Error.
    Formula: 200 * (|y_true - y_pred| / (|y_true| + |y_pred|))
    Uses epsilon to avoid division by zero.
    """
    y_t, y_p = to_numpy_arrays(y_true, y_pred)
    abs_diff = np.abs(y_t - y_p)
    sum_abs = np.abs(y_t) + np.abs(y_p)
    denom = np.where(sum_abs < epsilon, epsilon, sum_abs)
    return float(np.mean(2.0 * abs_diff / denom) * 100.0)

def calculate_pinball_loss(y_true: Any, y_pred: Any, quantile: float = 0.5) -> float:
    """
    Calculates Pinball Loss (Quantile Loss) for a given quantile.
    Formula: max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"Quantile must be in (0, 1) range, got: {quantile}")
        
    y_t, y_p = to_numpy_arrays(y_true, y_pred)
    diff = y_t - y_p
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(np.mean(loss))

def evaluate_all_metrics(
    y_true: Any,
    y_pred: Any,
    quantile_preds: Optional[Dict[float, Any]] = None
) -> Dict[str, float]:
    """
    Computes all standard forecast metrics (MAE, RMSE, MAPE, R2, SMAPE) 
    and optional Pinball Losses if quantile predictions are supplied.
    
    Parameters:
    y_true (array-like): Actual demand levels.
    y_pred (array-like): Point forecast predictions.
    quantile_preds (dict): Map from float quantile (e.g. 0.05, 0.95) to predictions array.
    
    Returns:
    Dict[str, float]: Compiled metrics dictionary.
    """
    metrics = {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "R2": calculate_r2(y_true, y_pred),
        "SMAPE": calculate_smape(y_true, y_pred)
    }
    
    if quantile_preds:
        for q, pred in quantile_preds.items():
            metrics[f"Pinball_Loss_{q}"] = calculate_pinball_loss(y_true, pred, quantile=q)
            
    return metrics
import typing
Any = typing.Any
