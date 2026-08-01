import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
from pathlib import Path
import shap

from src.explainability.shap_explainer import get_best_model_info, SHAPExplainerManager
from src.explainability.feature_importance import get_ranked_shap_feature_importance
from src.explainability.local_explanations import (
    identify_representative_samples, extract_local_explanation, generate_local_explanations_report
)
from src.explainability.visualization import (
    plot_shap_bar, plot_shap_summary, generate_all_shap_plots
)
from src.explainability.report_generator import generate_shap_report

class TestExplainability(unittest.TestCase):
    """Unit tests for Module 4 SHAP Explainability suite."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reports_dir = self.temp_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Mock best_model.json
        best_meta = {
            "best_model": "xgboost",
            "selection_criterion": "Lowest RMSE",
            "evaluation_metrics": {
                "MAE": 150.0,
                "RMSE": 210.0,
                "WMAPE": "26.50%",
                "R2": 0.72
            }
        }
        with open(self.reports_dir / "best_model.json", "w", encoding="utf-8") as f:
            json.dump(best_meta, f)

        # Synthetic feature data & SHAP values
        np.random.seed(42)
        self.feature_names = [f"Feature_{i}" for i in range(10)]
        self.X_test = pd.DataFrame(
            np.random.normal(10, 2, (30, 10)),
            columns=self.feature_names
        )

        self.shap_matrix = np.random.normal(0, 5, (30, 10))
        self.shap_exp = shap.Explanation(
            values=self.shap_matrix,
            base_values=np.full(30, 500.0),
            data=self.X_test.values,
            feature_names=self.feature_names
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_best_model_info(self):
        meta = get_best_model_info(self.reports_dir)
        self.assertEqual(meta["best_model"], "xgboost")
        self.assertIn("evaluation_metrics", meta)

    def test_get_ranked_shap_feature_importance(self):
        csv_path = self.reports_dir / "shap_feature_importance.csv"
        df_imp = get_ranked_shap_feature_importance(self.shap_exp, self.feature_names, save_path=csv_path)

        self.assertEqual(len(df_imp), 10)
        self.assertIn("Mean_Absolute_SHAP", df_imp.columns)
        self.assertIn("Percentage_Importance", df_imp.columns)
        self.assertTrue(csv_path.exists())

    def test_identify_representative_samples(self):
        y_pred = np.array([100.0, 50.0, 500.0, 250.0, 10.0])
        indices = identify_representative_samples(y_pred)

        self.assertEqual(indices["lowest"], 4)   # 10.0 (idx 4)
        self.assertEqual(indices["highest"], 2)  # 500.0 (idx 2)
        self.assertEqual(indices["median"], 0)   # 100.0 (sorted: 10, 50, 100, 250, 500 -> median element 100 is at orig idx 0)

    def test_extract_local_explanation(self):
        local_info = extract_local_explanation(
            shap_matrix=self.shap_matrix,
            feature_values=self.X_test,
            sample_idx=0,
            base_value=500.0,
            actual_val=520.0,
            pred_val=515.0
        )

        self.assertEqual(local_info["sample_index"], 0)
        self.assertEqual(local_info["base_value"], 500.0)
        self.assertIn("top_positive_features", local_info)
        self.assertIn("top_negative_features", local_info)

    def test_generate_local_explanations_report(self):
        rep_indices = identify_representative_samples(np.arange(30))
        local_dict = {
            k: extract_local_explanation(self.shap_matrix, self.X_test, idx, 500.0, 520.0, 515.0)
            for k, idx in rep_indices.items()
        }

        save_p = self.reports_dir / "local_explanations.md"
        generate_local_explanations_report(local_dict, "xgboost", save_p)

        self.assertTrue(save_p.exists())
        content = save_p.read_text(encoding="utf-8")
        self.assertIn("Local Forecast Explanations Report", content)

    def test_generate_shap_report(self):
        df_imp = get_ranked_shap_feature_importance(self.shap_exp, self.feature_names)
        rep_indices = identify_representative_samples(np.arange(30))
        local_dict = {
            k: extract_local_explanation(self.shap_matrix, self.X_test, idx, 500.0, 520.0, 515.0)
            for k, idx in rep_indices.items()
        }

        save_p = self.reports_dir / "shap_report.md"
        meta = get_best_model_info(self.reports_dir)
        generate_shap_report(meta, df_imp, local_dict, save_p)

        self.assertTrue(save_p.exists())
        content = save_p.read_text(encoding="utf-8")
        self.assertIn("SHAP Explainability & Model Transparency Report", content)

    def test_generate_all_shap_plots(self):
        df_imp = get_ranked_shap_feature_importance(self.shap_exp, self.feature_names)
        rep_indices = identify_representative_samples(np.arange(30))
        plots_dir = self.reports_dir / "shap_plots"

        generate_all_shap_plots(
            shap_explanation=self.shap_exp,
            shap_matrix=self.shap_matrix,
            X_test=self.X_test,
            expected_value=500.0,
            importance_df=df_imp,
            rep_indices=rep_indices,
            top_5_features=self.feature_names[:5],
            output_dir=plots_dir,
            dpi=100
        )

        expected_files = [
            "shap_summary.png",
            "shap_bar.png",
            "shap_waterfall_highest.png",
            "shap_waterfall_median.png",
            "shap_waterfall_lowest.png",
            "dependence_feature_1.png",
            "dependence_feature_2.png",
            "dependence_feature_3.png",
            "dependence_feature_4.png",
            "dependence_feature_5.png",
            "shap_force.html"
        ]

        for fname in expected_files:
            f_path = plots_dir / fname
            self.assertTrue(f_path.exists(), f"Missing SHAP plot artifact: {fname}")
            self.assertGreater(f_path.stat().st_size, 0, f"SHAP plot artifact empty: {fname}")

if __name__ == "__main__":
    unittest.main()
