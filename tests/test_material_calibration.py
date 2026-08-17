"""
Unit tests for src.evaluation.material_calibration module.
"""

import unittest
import pandas as pd
import numpy as np
from src.evaluation.material_calibration import (
    calculate_material_uncertainty_analysis,
    run_quantile_strategy_experiment
)

class TestMaterialCalibration(unittest.TestCase):
    """Test cases for material-level uncertainty calibration."""

    def setUp(self):
        self.df_sample = pd.DataFrame({
            "Date": ["2025-01-01"] * 6,
            "Material_Type": ["Conductor", "Conductor", "Hardware Fittings", "Hardware Fittings", "Transformer", "Transformer"],
            "Region": ["NR", "SR", "NR", "SR", "NR", "SR"],
            "Quantity_Required": [1000.0, 1200.0, 20.0, 10.0, 500.0, 600.0],
            "Forecast_Prediction_P10": [800.0, 900.0, 5.0, 2.0, 400.0, 450.0],
            "Forecast_Prediction": [1000.0, 1150.0, 18.0, 8.0, 480.0, 580.0],
            "Forecast_Prediction_P90": [1100.0, 1180.0, 25.0, 15.0, 550.0, 620.0]  # Row 1 (1200 > 1180 P90 violation)
        })

    def test_calculate_material_uncertainty_analysis(self):
        res_df = calculate_material_uncertainty_analysis(self.df_sample)
        self.assertFalse(res_df.empty)
        self.assertIn("Coefficient_of_Variation", res_df.columns)
        self.assertIn("PICP", res_df.columns)

    def test_run_quantile_strategy_experiment(self):
        mat_df = calculate_material_uncertainty_analysis(self.df_sample)
        comp_df = run_quantile_strategy_experiment(self.df_sample, mat_df)
        self.assertFalse(comp_df.empty)
        self.assertIn("Experiment A (P10-P50-P90 Baseline)", comp_df["Strategy"].values)
        self.assertIn("Experiment B (P05-P50-P95 Tailored)", comp_df["Strategy"].values)

if __name__ == "__main__":
    unittest.main()
