import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from src.evaluation.statistical_analysis import (
    rank_models,
    compute_wilcoxon_tests,
    compute_friedman_test,
    generate_statistical_report
)
from src.evaluation.publication_plots import (
    generate_all_publication_plots,
    plot_actual_vs_predicted_best,
    plot_probabilistic_forecast
)

class TestStatisticalAnalysis(unittest.TestCase):
    """Unit tests for statistical analysis, Wilcoxon, Friedman, and ranking logic."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Synthetic evaluation report DataFrame
        self.report_df = pd.DataFrame([
            {"Model": "xgboost", "MAE": 120.0, "RMSE": 180.0, "WMAPE": "25.0%", "R²": 0.75, "Training Time": 1.5, "Inference Time": 0.02},
            {"Model": "random_forest", "MAE": 130.0, "RMSE": 195.0, "WMAPE": "27.0%", "R²": 0.70, "Training Time": 2.0, "Inference Time": 0.05},
            {"Model": "lstm", "MAE": 110.0, "RMSE": 170.0, "WMAPE": "22.0%", "R²": 0.78, "Training Time": 5.0, "Inference Time": 0.10},
        ])

        # Synthetic prediction dictionary
        np.random.seed(42)
        y_actual = np.random.uniform(100, 1000, 100)

        self.preds_dict = {
            "xgboost": pd.DataFrame({
                "Quantity_Required": y_actual,
                "Forecast_Prediction": y_actual + np.random.normal(0, 50, 100)
            }),
            "random_forest": pd.DataFrame({
                "Quantity_Required": y_actual,
                "Forecast_Prediction": y_actual + np.random.normal(0, 60, 100)
            }),
            "lstm": pd.DataFrame({
                "Quantity_Required": y_actual,
                "Forecast_Prediction": y_actual + np.random.normal(0, 40, 100)
            })
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_rank_models(self):
        ranked = rank_models(self.report_df)
        self.assertEqual(len(ranked), 3)
        self.assertIn("Rank", ranked.columns)

        # First ranked model should be lstm (RMSE 170.0)
        top_model = ranked.iloc[0]["Model"]
        self.assertEqual(top_model, "lstm")
        self.assertEqual(ranked.iloc[0]["Rank"], 1)

    def test_compute_wilcoxon_tests(self):
        wilcoxon_df = compute_wilcoxon_tests(self.preds_dict, alpha=0.05)
        self.assertFalse(wilcoxon_df.empty)
        self.assertIn("Model_A", wilcoxon_df.columns)
        self.assertIn("Model_B", wilcoxon_df.columns)
        self.assertIn("P_Value", wilcoxon_df.columns)
        self.assertIn("Significant_Alpha_0.05", wilcoxon_df.columns)

    def test_compute_friedman_test(self):
        meta, friedman_df = compute_friedman_test(self.preds_dict, alpha=0.05)
        self.assertIn("Friedman_Statistic", meta)
        self.assertIn("P_Value", meta)
        self.assertEqual(meta["N_Models"], 3)
        self.assertFalse(friedman_df.empty)

    def test_generate_statistical_report(self):
        ranked = rank_models(self.report_df)
        wilcoxon_df = compute_wilcoxon_tests(self.preds_dict, alpha=0.05)
        meta, _ = compute_friedman_test(self.preds_dict, alpha=0.05)

        report_path = self.temp_dir / "statistical_evaluation.md"
        generate_statistical_report(ranked, wilcoxon_df, meta, report_path)

        self.assertTrue(report_path.exists())
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("Statistical Evaluation & Model Significance Report", content)
        self.assertIn("Friedman Test", content)
        self.assertIn("Wilcoxon Signed-Rank Test", content)


class TestPublicationPlots(unittest.TestCase):
    """Unit tests for publication-quality figure generation (300 DPI, IEEE compliance)."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.report_df = pd.DataFrame([
            {"Model": "xgboost", "MAE": 120.0, "RMSE": 180.0, "MAPE": 20.0, "WMAPE": 25.0, "SMAPE": 22.0, "R²": 0.75, "Training Time": 1.5, "Inference Time": 0.02},
            {"Model": "random_forest", "MAE": 130.0, "RMSE": 195.0, "MAPE": 22.0, "WMAPE": 27.0, "SMAPE": 24.0, "R²": 0.70, "Training Time": 2.0, "Inference Time": 0.05},
            {"Model": "lstm", "MAE": 110.0, "RMSE": 170.0, "MAPE": 18.0, "WMAPE": 22.0, "SMAPE": 20.0, "R²": 0.78, "Training Time": 5.0, "Inference Time": 0.10},
        ])

        np.random.seed(42)
        y_actual = np.random.uniform(100, 500, 50)
        self.best_preds = pd.DataFrame({
            "Quantity_Required": y_actual,
            "Forecast_Prediction": y_actual + np.random.normal(0, 15, 50)
        })

        self.quantile_preds = pd.DataFrame({
            "Quantity_Required": y_actual,
            "Forecast_Prediction_P10": y_actual - 30,
            "Forecast_Prediction": y_actual,
            "Forecast_Prediction_P90": y_actual + 30
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_generate_all_publication_plots(self):
        plots_dir = self.temp_dir / "model_plots"
        generate_all_publication_plots(
            report_df=self.report_df,
            best_preds_df=self.best_preds,
            quantile_preds_df=self.quantile_preds,
            output_dir=plots_dir,
            best_model_name="lstm",
            dpi=100
        )

        expected_plots = [
            "mae_comparison.png",
            "rmse_comparison.png",
            "mape_comparison.png",
            "wmape_comparison.png",
            "smape_comparison.png",
            "r2_comparison.png",
            "training_time_comparison.png",
            "inference_time_comparison.png",
            "actual_vs_predicted_best_model.png",
            "lightgbm_quantile_probabilistic_forecast.png"
        ]

        for p_name in expected_plots:
            p_file = plots_dir / p_name
            self.assertTrue(p_file.exists(), f"Plot missing: {p_name}")
            self.assertGreater(p_file.stat().st_size, 0, f"Plot file empty: {p_name}")


if __name__ == "__main__":
    unittest.main()
