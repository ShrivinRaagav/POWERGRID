import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.config.settings import REPORTS_DIR, EXPERIMENTS_DIR
from src.utils.helpers import setup_logger
from src.explainability.shap_explainer import SHAPExplainerManager
from src.explainability.feature_importance import get_ranked_shap_feature_importance
from src.explainability.local_explanations import (
    identify_representative_samples, extract_local_explanation, generate_local_explanations_report
)
from src.explainability.visualization import generate_all_shap_plots
from src.explainability.report_generator import generate_shap_report

logger = setup_logger("run_explainability")

def run_explainability_pipeline(reports_dir: Path = REPORTS_DIR) -> Dict[str, Any]:
    """
    Master pipeline orchestrator for Module 4 (SHAP Explainability):
    1. Loads best_model.json metadata & checkpoint.
    2. Computes SHAP values on the test dataset.
    3. Exports reports/shap_feature_importance.csv.
    4. Extracts Highest, Median, and Lowest demand predictions & exports local_explanations.md.
    5. Renders 10 publication-quality figures in reports/shap_plots/.
    6. Compiles reports/shap_report.md.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = reports_dir / "shap_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Module 4 SHAP Explainability Pipeline...")

    # 1. Initialize Explainer Manager & compute SHAP values
    manager = SHAPExplainerManager(reports_dir=reports_dir)
    shap_exp = manager.compute_shap_values()

    best_meta = manager.best_meta
    best_model_name = manager.best_model_name
    shap_matrix = manager.shap_matrix
    expected_val = manager.expected_value
    X_test = getattr(manager, "target_test", manager.X_test.iloc[:len(shap_matrix)])
    y_test = manager.y_test
    feature_names = manager.feature_names

    # 2. Global Feature Importance
    csv_path = reports_dir / "shap_feature_importance.csv"
    importance_df = get_ranked_shap_feature_importance(shap_exp, feature_names, save_path=csv_path)

    # 3. Model Predictions on Evaluated Test Set
    y_pred = manager._predict_array(X_test)

    y_pred_arr = np.asarray(y_pred).flatten()
    y_true_raw = y_test.values if hasattr(y_test, "values") else np.asarray(y_test).flatten()
    y_true_arr = y_true_raw[:len(X_test)]

    # 4. Representative Local Explanations
    rep_indices = identify_representative_samples(y_pred_arr)
    local_dict: Dict[str, Dict[str, Any]] = {}

    for scenario_name, sample_idx in rep_indices.items():
        act_v = float(y_true_arr[sample_idx]) if len(y_true_arr) > sample_idx else None
        pred_v = float(y_pred_arr[sample_idx])

        local_info = extract_local_explanation(
            shap_matrix=shap_matrix,
            feature_values=X_test,
            sample_idx=sample_idx,
            base_value=expected_val,
            actual_val=act_v,
            pred_val=pred_v
        )
        local_dict[scenario_name] = local_info

    generate_local_explanations_report(
        local_dict=local_dict,
        model_name=best_model_name,
        save_path=reports_dir / "local_explanations.md"
    )

    # 5. Render All SHAP Visualizations (Summary, Bar, Waterfalls, 5 Dependence, Force HTML)
    top_5_features = importance_df.head(5)["Feature"].tolist() if not importance_df.empty else feature_names[:5]

    generate_all_shap_plots(
        shap_explanation=shap_exp,
        shap_matrix=shap_matrix,
        X_test=X_test,
        expected_value=expected_val,
        importance_df=importance_df,
        rep_indices=rep_indices,
        top_5_features=top_5_features,
        output_dir=plots_dir,
        dpi=300
    )

    # 6. Export SHAP Comprehensive Report
    generate_shap_report(
        best_meta=best_meta,
        importance_df=importance_df,
        local_dict=local_dict,
        save_path=reports_dir / "shap_report.md"
    )

    logger.info("Module 4 SHAP Explainability Pipeline completed successfully.")
    return {
        "best_model": best_model_name,
        "importance_df": importance_df,
        "local_dict": local_dict
    }

def main():
    parser = argparse.ArgumentParser(description="POWERGRID SHAP Explainability Controller CLI (Module 4)")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory to save SHAP reports.")
    args = parser.parse_args()

    run_explainability_pipeline(reports_dir=Path(args.reports_dir))

if __name__ == "__main__":
    main()

import numpy as np
