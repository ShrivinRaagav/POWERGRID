"""
Unit tests for src.evaluation.uncertainty_metrics module.
"""

import unittest
import numpy as np
import pandas as pd
from src.evaluation.uncertainty_metrics import (
    calculate_picp,
    calculate_interval_widths,
    check_quantile_crossings,
    detect_interval_outliers
)

class TestUncertaintyMetrics(unittest.TestCase):
    """Test cases for uncertainty metrics and calibration analysis."""

    def test_calculate_picp(self):
        y_true = np.array([100, 200, 300, 400, 500])
        p10 = np.array([90, 190, 290, 450, 490])
        p90 = np.array([110, 210, 310, 500, 510])
        
        covered, total, picp_pct = calculate_picp(y_true, p10, p90)
        self.assertEqual(total, 5)
        self.assertEqual(covered, 4) # 4 out of 5 covered (400 is not between 450 and 500)
        self.assertAlmostEqual(picp_pct, 80.0)

    def test_calculate_interval_widths(self):
        p10 = np.array([100, 200, 300])
        p90 = np.array([150, 300, 450])
        
        avg_w, min_w, max_w = calculate_interval_widths(p10, p90)
        self.assertEqual(min_w, 50.0)
        self.assertEqual(max_w, 150.0)
        self.assertEqual(avg_w, 100.0)

    def test_check_quantile_crossings(self):
        df = pd.DataFrame({
            "P10": [10, 20, 30],
            "P50": [15, 18, 35], # Row index 1 has P10=20 > P50=18 (Crossing violation)
            "P90": [20, 25, 40]
        })
        crossing_count, viol_df = check_quantile_crossings(df)
        self.assertEqual(crossing_count, 1)

    def test_detect_interval_outliers(self):
        df = pd.DataFrame({
            "Date": ["2025-01-01", "2025-01-08", "2025-01-15"],
            "Material": ["Conductor", "Transformer", "Insulator"],
            "Region": ["NR", "SR", "WR"],
            "Actual_Demand": [100.0, 500.0, 50.0],
            "P10": [80.0, 100.0, 60.0],  # Row 1 (500 > 300 OVER_P90), Row 2 (50 < 60 UNDER_P10)
            "P50": [100.0, 200.0, 80.0],
            "P90": [120.0, 300.0, 100.0]
        })
        outliers_df = detect_interval_outliers(df)
        self.assertEqual(len(outliers_df), 2)
        self.assertIn("OVER_P90", outliers_df["Outside_Bound"].values)
        self.assertIn("UNDER_P10", outliers_df["Outside_Bound"].values)

if __name__ == "__main__":
    unittest.main()
