import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional
import shap

from src.utils.helpers import setup_logger

logger = setup_logger("shap_feature_importance")

def get_ranked_shap_feature_importance(
    shap_values: Union[shap.Explanation, np.ndarray],
    feature_names: List[str],
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Computes global feature importance based on mean absolute SHAP values across test samples.
    
    Parameters:
    shap_values: SHAP Explanation object or 2D numpy array of SHAP values.
    feature_names: List of feature names matching columns.
    save_path: Optional path to export CSV (reports/shap_feature_importance.csv).
    
    Returns:
    pd.DataFrame: Sorted feature importances table.
    """
    if isinstance(shap_values, shap.Explanation):
        vals = shap_values.values
    else:
        vals = np.asarray(shap_values)

    if vals.ndim > 2:
        vals = np.mean(vals, axis=-1)

    mean_abs_shap = np.mean(np.abs(vals), axis=0)

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_abs_shap
    })

    # Sort descending
    df_imp = df_imp.sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

    total_importance = df_imp["Mean_Absolute_SHAP"].sum()
    if total_importance > 1e-12:
        df_imp["Percentage_Importance"] = (df_imp["Mean_Absolute_SHAP"] / total_importance) * 100.0
    else:
        df_imp["Percentage_Importance"] = 0.0

    df_imp["Cumulative_Importance"] = df_imp["Percentage_Importance"].cumsum()
    df_imp["Rank"] = np.arange(1, len(df_imp) + 1)

    # Reorder columns
    cols = ["Rank", "Feature", "Mean_Absolute_SHAP", "Percentage_Importance", "Cumulative_Importance"]
    df_imp = df_imp[cols]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_imp.to_csv(save_path, index=False)
        logger.info(f"Exported SHAP feature importance table to {save_path}")

    return df_imp
