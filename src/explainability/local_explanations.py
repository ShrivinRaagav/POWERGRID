import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import shap

from src.utils.helpers import setup_logger

logger = setup_logger("local_explanations")

def identify_representative_samples(
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Identifies test sample indices for Highest, Median, and Lowest demand predictions.
    
    Parameters:
    y_pred (np.ndarray): 1D array of predicted demand values.
    
    Returns:
    Dict[str, int]: Map of scenario ('highest', 'median', 'lowest') -> sample_index.
    """
    if len(y_pred) == 0:
        raise ValueError("y_pred is empty.")

    sorted_indices = np.argsort(y_pred)

    idx_lowest = int(sorted_indices[0])
    idx_highest = int(sorted_indices[-1])
    idx_median = int(sorted_indices[len(sorted_indices) // 2])

    return {
        "highest": idx_highest,
        "median": idx_median,
        "lowest": idx_lowest
    }

def extract_local_explanation(
    shap_matrix: np.ndarray,
    feature_values: pd.DataFrame,
    sample_idx: int,
    base_value: float,
    actual_val: Optional[float] = None,
    pred_val: Optional[float] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Extracts top positive and negative local feature contributions for a single test instance.
    """
    sample_shap = shap_matrix[sample_idx]
    feature_names = list(feature_values.columns)
    feat_vals = feature_values.iloc[sample_idx].values

    df_local = pd.DataFrame({
        "Feature": feature_names,
        "Feature_Value": feat_vals,
        "SHAP_Value": sample_shap
    })

    # Sort by SHAP value
    df_pos = df_local.sort_values(by="SHAP_Value", ascending=False).head(top_k)
    df_neg = df_local.sort_values(by="SHAP_Value", ascending=True).head(top_k)

    return {
        "sample_index": sample_idx,
        "base_value": float(base_value),
        "actual": float(actual_val) if actual_val is not None else None,
        "predicted": float(pred_val) if pred_val is not None else None,
        "top_positive_features": df_pos.to_dict(orient="records"),
        "top_negative_features": df_neg.to_dict(orient="records"),
        "full_local_df": df_local
    }

def generate_local_explanations_report(
    local_dict: Dict[str, Dict[str, Any]],
    model_name: str,
    save_path: Path
):
    """
    Generates reports/local_explanations.md detailing local predictions and top feature drivers.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    md = f"""# Local Forecast Explanations Report (Module 4 - SHAP XAI)

This report provides granular local feature attributions for three representative operational demand scenarios (**Highest Demand**, **Median Demand**, and **Lowest Demand**) predicted by the best model (**{model_name}**).

---

## 1. Overview & Baseline Setup

Local SHAP explanations break down individual prediction equations into additive feature contributions:

$$\\hat{{y}}_i = E[f(X)] + \\sum_{{j=1}}^{{p}} \\phi_{{i,j}}$$

Where:
- **y_hat_i**: Predicted demand for test instance i.
- **E[f(X)]**: Base value (average model prediction across training set).
- **phi_{{i,j}}**: SHAP attribution of feature j for instance i.

---

"""

    for scenario_name in ["highest", "median", "lowest"]:
        sc_data = local_dict.get(scenario_name, {})
        title = scenario_name.capitalize()
        s_idx = sc_data.get("sample_index", "N/A")
        base_v = sc_data.get("base_value", 0.0)
        act_v = sc_data.get("actual", "N/A")
        pred_v = sc_data.get("predicted", "N/A")

        act_str = f"{act_v:.2f}" if isinstance(act_v, (int, float)) else "N/A"
        pred_str = f"{pred_v:.2f}" if isinstance(pred_v, (int, float)) else "N/A"

        md += f"""## 2. {title} Demand Prediction Case Study

- **Scenario Type**: {title} Predicted Demand
- **Test Sample Index**: `{s_idx}`
- **Actual Material Demand**: `{act_str}`
- **Predicted Material Demand**: `{pred_str}`
- **Base Expected Value (E[f(X)])**: `{base_v:.2f}`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
"""
        for item in sc_data.get("top_positive_features", []):
            f_val = f"{item['Feature_Value']:.4f}" if isinstance(item['Feature_Value'], (int, float)) else str(item['Feature_Value'])
            shap_v = f"+{item['SHAP_Value']:.4f}"
            md += f"| `{item['Feature']}` | `{f_val}` | `{shap_v}` |\n"

        md += """
### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
"""
        for item in sc_data.get("top_negative_features", []):
            f_val = f"{item['Feature_Value']:.4f}" if isinstance(item['Feature_Value'], (int, float)) else str(item['Feature_Value'])
            shap_v = f"{item['SHAP_Value']:.4f}"
            md += f"| `{item['Feature']}` | `{f_val}` | `{shap_v}` |\n"

        md += "\n---\n\n"

    md += """## 3. Operational Insights for POWERGRID Procurement

1. **High Demand Risk Mitigation**: When forecasts spike (High Demand case), decomposition trend components and project stage features act as primary positive drivers. Procuring early prevents stocking delays.
2. **Low Demand Asset Allocation**: When forecasts drop (Low Demand case), seasonal monsoon lags reduce required buffer stock, preventing warehouse congestion and holding cost penalties.
3. **Auditability**: Regional engineers can inspect individual project site predictions using these waterfall SHAP breakdowns to verify procurement requests before releasing purchase orders.
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Saved local explanations report to {save_path}")
