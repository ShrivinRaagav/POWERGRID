import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import os

from src.evaluation.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    calculate_r2,
    calculate_smape,
    calculate_pinball_loss,
    evaluate_all_metrics
)
from src.evaluation.comparison import ModelComparisonRegistry
from src.evaluation.visualization import (
    plot_predictions_vs_actuals,
    plot_residuals_distribution,
    plot_prediction_intervals
)

class TestMetrics(unittest.TestCase):
    """Tests the calculation accuracy and numerical safety of the validation metrics."""
    def setUp(self):
        self.y_true = np.array([100.0, 200.0, 300.0])
        self.y_pred = np.array([110.0, 190.0, 300.0]) # Errors: 10, -10, 0
        
    def test_calculate_mae(self):
        mae = calculate_mae(self.y_true, self.y_pred)
        # (|10| + |-10| + 0) / 3 = 20 / 3 = 6.6666...
        self.assertAlmostEqual(mae, 20.0 / 3.0)

    def test_calculate_rmse(self):
        rmse = calculate_rmse(self.y_true, self.y_pred)
        # sqrt((100 + 100 + 0) / 3) = sqrt(200 / 3) = 8.1649...
        self.assertAlmostEqual(rmse, np.sqrt(200.0 / 3.0))

    def test_calculate_mape(self):
        mape = calculate_mape(self.y_true, self.y_pred)
        # (|10/100| + |-10/200| + 0) / 3 * 100 = (0.1 + 0.05) / 3 * 100 = 5.0%
        self.assertAlmostEqual(mape, 5.0)

    def test_calculate_mape_zero_handling(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([10.0, 100.0])
        mape = calculate_mape(y_true, y_pred)
        # Zero true value should be replaced by epsilon (1e-5), so error is 10 / 1e-5 = 1e6
        self.assertGreater(mape, 10000.0)

    def test_calculate_r2(self):
        r2 = calculate_r2(self.y_true, self.y_pred)
        self.assertLess(r2, 1.0)
        self.assertGreater(r2, 0.0)

    def test_calculate_smape(self):
        smape = calculate_smape(self.y_true, self.y_pred)
        # 200 * [ (10 / (100+110)) + (10 / (200+190)) + 0 ] / 3 = 200 * [10/210 + 10/390] / 3
        expected = 2.0 * ((10.0 / 210.0) + (10.0 / 390.0)) / 3.0 * 100.0
        self.assertAlmostEqual(smape, expected)

    def test_calculate_pinball_loss(self):
        y_true = np.array([10.0])
        y_pred = np.array([12.0]) # diff = -2.0
        
        # Quantile 0.1: max(0.1 * -2, -0.9 * -2) = max(-0.2, 1.8) = 1.8
        loss_10 = calculate_pinball_loss(y_true, y_pred, quantile=0.1)
        self.assertAlmostEqual(loss_10, 1.8)
        
        # Quantile 0.9: max(0.9 * -2, -0.1 * -2) = max(-1.8, 0.2) = 0.2
        loss_90 = calculate_pinball_loss(y_true, y_pred, quantile=0.9)
        self.assertAlmostEqual(loss_90, 0.2)

    def test_evaluate_all_metrics(self):
        quantile_preds = {
            0.1: np.array([90.0, 180.0, 270.0]),
            0.9: np.array([120.0, 220.0, 320.0])
        }
        res = evaluate_all_metrics(self.y_true, self.y_pred, quantile_preds)
        self.assertIn("MAE", res)
        self.assertIn("RMSE", res)
        self.assertIn("Pinball_Loss_0.1", res)
        self.assertIn("Pinball_Loss_0.9", res)


class TestComparisonAndPlots(unittest.TestCase):
    """Tests the registry list and validation plotting tools."""
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.y_true = np.sin(np.linspace(0, 10, 100)) + 10.0
        self.y_pred = self.y_true + np.random.normal(0, 0.5, 100)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_comparison_registry(self):
        registry = ModelComparisonRegistry()
        
        m1_metrics = {"MAE": 2.5, "RMSE": 3.0, "R2": 0.85}
        m2_metrics = {"MAE": 1.5, "RMSE": 2.0, "R2": 0.92}
        
        registry.register_model("Model_A", m1_metrics)
        registry.register_model("Model_B", m2_metrics)
        
        # Best model rank
        best = registry.get_best_model(rank_by_metric="MAE", ascending=True)
        self.assertEqual(best, "Model_B")
        
        best_r2 = registry.get_best_model(rank_by_metric="R2", ascending=False)
        self.assertEqual(best_r2, "Model_B")
        
        # Save check
        csv_path = self.temp_dir / "comparison.csv"
        registry.save_comparison_csv(csv_path)
        self.assertTrue(csv_path.exists())
        
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)
        self.assertIn("Model_Name", df.columns)

    def test_plots_generation(self):
        p1 = self.temp_dir / "pred_vs_act.png"
        p2 = self.temp_dir / "residuals.png"
        p3 = self.temp_dir / "intervals.png"
        
        plot_predictions_vs_actuals(self.y_true, self.y_pred, p1, dpi=100)
        plot_residuals_distribution(self.y_true, self.y_pred, p2, dpi=100)
        
        lower = self.y_pred - 1.0
        upper = self.y_pred + 1.0
        plot_prediction_intervals(self.y_true, self.y_pred, lower, upper, p3, dpi=100)
        
        self.assertTrue(p1.exists())
        self.assertTrue(p2.exists())
        self.assertTrue(p3.exists())

if __name__ == "__main__":
    unittest.main()
