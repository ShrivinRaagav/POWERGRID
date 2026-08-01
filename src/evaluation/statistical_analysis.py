import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import wilcoxon, friedmanchisquare

from src.utils.helpers import setup_logger

logger = setup_logger("statistical_analysis")

def rank_models(report_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks production models based on RMSE (primary), MAE (secondary), and WMAPE (tertiary).
    
    Parameters:
    report_df (pd.DataFrame): DataFrame containing columns 'Model', 'RMSE', 'MAE', 'WMAPE', 'R²'.
    
    Returns:
    pd.DataFrame: Ranked models DataFrame with a 'Rank' column (1-indexed).
    """
    if report_df.empty:
        logger.warning("Empty report DataFrame provided to rank_models.")
        return pd.DataFrame(columns=["Rank", "Model", "RMSE", "MAE", "WMAPE", "R²"])

    df = report_df.copy()

    # Clean numerical columns
    for col in ["RMSE", "MAE", "WMAPE", "R²"]:
        if col in df.columns:
            if df[col].dtype == object:
                # Remove % if present
                df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort primarily by RMSE, secondarily by MAE, tertiarily by WMAPE
    sort_cols = [c for c in ["RMSE", "MAE", "WMAPE"] if c in df.columns]
    df = df.sort_values(by=sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)

    df["Rank"] = np.arange(1, len(df) + 1)
    output_cols = [c for c in ["Rank", "Model", "RMSE", "MAE", "WMAPE", "R²"] if c in df.columns]

    logger.info(f"Successfully ranked {len(df)} models.")
    return df[output_cols]

def compute_wilcoxon_tests(
    predictions_dict: Dict[str, pd.DataFrame],
    target_col: str = "Quantity_Required",
    pred_col: str = "Forecast_Prediction",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Performs pairwise Wilcoxon Signed-Rank tests on absolute prediction errors (|y - y_hat|)
    between all pairs of production models on identical test samples.
    
    Parameters:
    predictions_dict (dict): Dictionary mapping model_name -> DataFrame containing target and prediction.
    target_col (str): Column name for actual target values.
    pred_col (str): Column name for model point predictions.
    alpha (float): Significance threshold (default 0.05).
    
    Returns:
    pd.DataFrame: Pairwise Wilcoxon test results.
    """
    model_names = sorted(list(predictions_dict.keys()))
    if len(model_names) < 2:
        logger.warning("Fewer than 2 models provided for Wilcoxon pairwise test.")
        return pd.DataFrame(columns=[
            "Model_A", "Model_B", "Statistic", "P_Value",
            "Significant_Alpha_0.05", "Mean_AE_Model_A", "Mean_AE_Model_B", "Superior_Model"
        ])

    # Extract absolute errors per model
    abs_errors: Dict[str, np.ndarray] = {}
    for name in model_names:
        df = predictions_dict[name]
        if target_col in df.columns and pred_col in df.columns:
            err = np.abs(df[target_col].values - df[pred_col].values)
            abs_errors[name] = err
        else:
            logger.warning(f"Model '{name}' predictions missing required columns.")

    valid_models = list(abs_errors.keys())
    results = []

    for i in range(len(valid_models)):
        for j in range(i + 1, len(valid_models)):
            m_a = valid_models[i]
            m_b = valid_models[j]

            err_a = abs_errors[m_a]
            err_b = abs_errors[m_b]

            # Ensure matching lengths
            n_samples = min(len(err_a), len(err_b))
            err_a = err_a[:n_samples]
            err_b = err_b[:n_samples]

            diff = err_a - err_b
            mae_a = float(np.mean(err_a))
            mae_b = float(np.mean(err_b))

            if np.allclose(diff, 0.0):
                stat, p_val = 0.0, 1.0
            else:
                try:
                    res = wilcoxon(err_a, err_b, zero_method="pratt")
                    stat, p_val = float(res.statistic), float(res.pvalue)
                except Exception as e:
                    logger.warning(f"Wilcoxon test failed for {m_a} vs {m_b}: {e}")
                    stat, p_val = np.nan, np.nan

            is_sig = bool(p_val < alpha) if not np.isnan(p_val) else False

            if is_sig:
                superior = m_a if mae_a < mae_b else m_b
            else:
                superior = "No Statistically Significant Difference"

            results.append({
                "Model_A": m_a,
                "Model_B": m_b,
                "Statistic": stat,
                "P_Value": p_val,
                "Significant_Alpha_0.05": is_sig,
                "Mean_AE_Model_A": mae_a,
                "Mean_AE_Model_B": mae_b,
                "Superior_Model": superior
            })

    res_df = pd.DataFrame(results)
    logger.info(f"Completed Wilcoxon signed-rank tests for {len(results)} pairwise comparisons.")
    return res_df

def compute_friedman_test(
    predictions_dict: Dict[str, pd.DataFrame],
    target_col: str = "Quantity_Required",
    pred_col: str = "Forecast_Prediction",
    alpha: float = 0.05
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Performs the Friedman non-parametric test across all production models on absolute prediction errors.
    
    Parameters:
    predictions_dict (dict): Dictionary mapping model_name -> DataFrame containing predictions.
    target_col (str): Target column name.
    pred_col (str): Prediction column name.
    alpha (float): Significance threshold (default 0.05).
    
    Returns:
    Tuple[Dict[str, Any], pd.DataFrame]: (Friedman test metadata dict, Friedman results DataFrame).
    """
    model_names = sorted(list(predictions_dict.keys()))
    if len(model_names) < 2:
        logger.warning("Fewer than 2 models provided for Friedman test.")
        meta = {
            "Friedman_Statistic": np.nan,
            "P_Value": np.nan,
            "Degrees_Of_Freedom": 0,
            "Significant_Alpha_0.05": False,
            "N_Samples": 0,
            "N_Models": len(model_names)
        }
        df_res = pd.DataFrame([meta])
        return meta, df_res

    err_arrays = []
    valid_names = []
    min_len = float("inf")

    for name in model_names:
        df = predictions_dict[name]
        if target_col in df.columns and pred_col in df.columns:
            err = np.abs(df[target_col].values - df[pred_col].values)
            err_arrays.append(err)
            valid_names.append(name)
            if len(err) < min_len:
                min_len = len(err)

    if len(valid_names) < 2:
        meta = {
            "Friedman_Statistic": np.nan,
            "P_Value": np.nan,
            "Degrees_Of_Freedom": 0,
            "Significant_Alpha_0.05": False,
            "N_Samples": 0,
            "N_Models": len(valid_names)
        }
        return meta, pd.DataFrame([meta])

    # Truncate to min_len to ensure aligned samples
    aligned_errs = [err[:min_len] for err in err_arrays]

    try:
        stat, p_val = friedmanchisquare(*aligned_errs)
        stat, p_val = float(stat), float(p_val)
    except Exception as e:
        logger.error(f"Friedman test calculation failed: {e}")
        stat, p_val = np.nan, np.nan

    df_degrees = len(valid_names) - 1
    is_sig = bool(p_val < alpha) if not np.isnan(p_val) else False

    meta = {
        "Friedman_Statistic": stat,
        "P_Value": p_val,
        "Degrees_Of_Freedom": df_degrees,
        "Significant_Alpha_0.05": is_sig,
        "N_Samples": min_len,
        "N_Models": len(valid_names)
    }

    res_df = pd.DataFrame([meta])
    logger.info(f"Friedman Test: Stat={stat:.4f}, p-value={p_val:.6e}, Significant={is_sig}")
    return meta, res_df

def generate_statistical_report(
    ranking_df: pd.DataFrame,
    wilcoxon_df: pd.DataFrame,
    friedman_meta: Dict[str, Any],
    save_path: Path
):
    """
    Compiles reports/statistical_evaluation.md detailing the statistical evaluation findings.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    stat_val = friedman_meta.get("Friedman_Statistic", np.nan)
    p_val = friedman_meta.get("P_Value", np.nan)
    is_sig = friedman_meta.get("Significant_Alpha_0.05", False)
    dof = friedman_meta.get("Degrees_Of_Freedom", 0)
    n_samples = friedman_meta.get("N_Samples", 0)
    n_models = friedman_meta.get("N_Models", 0)

    p_val_str = f"{p_val:.6e}" if pd.notnull(p_val) else "N/A"
    stat_str = f"{stat_val:.4f}" if pd.notnull(stat_val) else "N/A"

    md_content = f"""# Statistical Evaluation & Model Significance Report (Module 3.5)

This report presents the rigorous statistical evaluation and pairwise significance analysis for the production forecasting models trained on the POWERGRID transmission material demand dataset.

---

## 1. Executive Summary

- **Primary Ranking Metric**: Root Mean Squared Error (RMSE)
- **Secondary Ranking Metrics**: Mean Absolute Error (MAE) & Weighted Mean Absolute Percentage Error (WMAPE)
- **Significance Level ($\alpha$)**: 0.05
- **Overall Multi-Model Comparison**: Friedman Non-Parametric Test
- **Pairwise Model Comparison**: Wilcoxon Signed-Rank Test

---

## 2. Overall Multi-Model Comparison (Friedman Test)

The **Friedman Test** evaluates whether there is a statistically significant difference in performance across all evaluated forecasting models.

| Metric | Value |
| :--- | :--- |
| **Friedman $\chi^2$ Statistic** | `{stat_str}` |
| **p-value** | `{p_val_str}` |
| **Degrees of Freedom** | `{dof}` |
| **Number of Evaluated Models** | `{n_models}` |
| **Test Set Sample Size** | `{n_samples}` |
| **Statistically Significant ($\alpha=0.05$)** | **{"YES" if is_sig else "NO"}** |

> [!NOTE]
> {"The p-value is below 0.05, confirming statistically significant performance differences among the models across the chronological test set." if is_sig else "No statistically significant overall variance was detected across models at alpha = 0.05."}

---

## 3. Pairwise Model Significance (Wilcoxon Signed-Rank Test)

The **Wilcoxon Signed-Rank Test** performs non-parametric pairwise comparisons on absolute prediction errors ($|y - \hat{{y}}|$) across models.

| Model A | Model B | Statistic | p-value | Significant ($\alpha=0.05$) | Superior Model |
| :--- | :--- | :---: | :---: | :---: | :--- |
"""

    if not wilcoxon_df.empty:
        for _, row in wilcoxon_df.iterrows():
            w_stat = f"{row['Statistic']:.2f}" if pd.notnull(row['Statistic']) else "N/A"
            w_p = f"{row['P_Value']:.6e}" if pd.notnull(row['P_Value']) else "N/A"
            sig_str = "YES" if row['Significant_Alpha_0.05'] else "NO"
            sup_model = str(row['Superior_Model'])
            md_content += f"| {row['Model_A']} | {row['Model_B']} | {w_stat} | {w_p} | {sig_str} | {sup_model} |\n"
    else:
        md_content += "| N/A | N/A | N/A | N/A | N/A | Insufficient Model Runs |\n"

    md_content += """
---

## 4. Official Model Performance Ranking

Models are ranked strictly by **RMSE (ascending)** as the primary criterion, followed by **MAE** and **WMAPE**.

| Rank | Model | RMSE | MAE | WMAPE (%) | R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""

    if not ranking_df.empty:
        for _, row in ranking_df.iterrows():
            rmse_v = f"{row['RMSE']:.6f}" if pd.notnull(row['RMSE']) else "N/A"
            mae_v = f"{row['MAE']:.6f}" if pd.notnull(row['MAE']) else "N/A"
            wmape_v = f"{row['WMAPE']:.2f}%" if pd.notnull(row['WMAPE']) else "N/A"
            r2_v = f"{row['R²']:.6f}" if pd.notnull(row['R²']) else "N/A"
            md_content += f"| {row['Rank']} | {row['Model']} | {rmse_v} | {mae_v} | {wmape_v} | {r2_v} |\n"

    md_content += """
---

## 5. Methodological Notes

1. **Non-Parametric Assumptions**: Forecast error distributions in supply chain demand are non-Gaussian due to monsoon seasonality and site disruptions. Therefore, non-parametric Wilcoxon and Friedman tests are mandated.
2. **Error Metric Definition**: Absolute Error $e_{i,m} = |y_i - \hat{y}_{i,m}|$ is calculated per time-series sample $i$ for model $m$.
3. **Downstream Integration**: The rank-1 model is selected as `best_model.json` and will be forwarded to **Module 4 (SHAP Explainability)** and **Module 5 (Multi-Objective Optimization)**.
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Saved statistical evaluation report to {save_path}")
