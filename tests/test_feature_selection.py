import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import os

from src.feature_selection.variance import find_low_variance_features
from src.feature_selection.correlation import find_highly_correlated_features
from src.feature_selection.importance import calculate_feature_importances
from src.feature_selection.utils import find_duplicated_features
from src.feature_selection.feature_selector import FeatureSelector

class TestFeatureSelectionComponents(unittest.TestCase):
    """Tests the standalone filtering functions in the feature selection package."""
    def setUp(self):
        # Create a mock dataframe for testing individual filters
        self.df = pd.DataFrame({
            "Constant": [5.0] * 10,                    # Var = 0
            "LowVar": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.05], # Low var
            "NormalVar": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], # High var
            "Duplicate": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], # Identical to NormalVar
            "Correlated": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1], # Correlated with NormalVar (~0.99)
            "Target": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], # Linear relation to NormalVar
            "Protected": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        })

    def test_find_duplicated_features(self):
        dups = find_duplicated_features(self.df, exclude_cols=["Protected"])
        self.assertIn("Duplicate", dups)
        self.assertNotIn("NormalVar", dups)

    def test_find_low_variance_features(self):
        low_var_cols, variances = find_low_variance_features(
            self.df,
            threshold=0.01,
            exclude_cols=["Protected"]
        )
        self.assertIn("Constant", low_var_cols)
        self.assertIn("LowVar", low_var_cols)
        self.assertNotIn("NormalVar", low_var_cols)

    def test_find_highly_correlated_features(self):
        # We simulate importances: NormalVar is more important than Correlated
        importances = {"NormalVar": 0.8, "Correlated": 0.2}
        variances = self.df.var()
        
        # Keep Protected, Constant, and LowVar excluded to test correlation isolated
        drop_cols, corr_matrix, pair_details = find_highly_correlated_features(
            df=self.df,
            threshold=0.85,
            exclude_cols=["Protected", "Constant", "LowVar", "Duplicate", "Target"],
            importances=importances,
            variances=variances
        )
        # Correlated has lower importance, so it should be dropped
        self.assertIn("Correlated", drop_cols)
        self.assertNotIn("NormalVar", drop_cols)
        self.assertGreater(len(pair_details), 0)

    def test_calculate_feature_importances(self):
        importances = calculate_feature_importances(
            df=self.df,
            target_col="Target",
            exclude_cols=["Protected", "Constant", "LowVar", "Duplicate", "Correlated"],
            method="mutual_info"
        )
        # NormalVar is linearly related to Target, so it should have strong importance
        self.assertIn("NormalVar", importances)
        self.assertGreater(importances["NormalVar"], 0.0)


class TestFeatureSelector(unittest.TestCase):
    """Tests the orchestrating FeatureSelector class."""
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.save_path = self.temp_dir / "feature_selector.joblib"
        self.corr_path = self.temp_dir / "corr.joblib"
        self.report_path = self.temp_dir / "report.csv"
        
        t = np.linspace(0, 2, 100)
        self.df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=100, freq="W"),
            "Project_ID": ["PG-1"] * 100,
            "Region": ["NR"] * 100,
            "Material_Type": ["Conductor"] * 100,
            "Constant": [2.5] * 100,
            "Noise_1": np.random.normal(0, 1, 100),
            "Noise_2": np.random.normal(0, 1, 100),
            "Signal": np.sin(t) * 100.0 + np.random.normal(0, 0.1, 100),
            "Duplicate": np.sin(t) * 100.0 + np.random.normal(0, 0.1, 100), # Identical or very similar
            "Quantity_Required": np.sin(t) * 100.0
        })
        # Force duplicate to be exactly identical
        self.df["Duplicate"] = self.df["Signal"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_fit_transform_pipeline(self):
        selector = FeatureSelector(
            variance_threshold=0.01,
            correlation_threshold=0.85,
            importance_method="mutual_info",
            top_k=2,
            keep_cols=["Date", "Project_ID", "Region", "Material_Type"],
            save_path=self.save_path,
            corr_save_path=self.corr_path,
            report_path=self.report_path
        )
        
        selector.fit(self.df, target_col="Quantity_Required")
        
        # Verify fitted state
        self.assertTrue(selector.is_fit)
        self.assertTrue(self.save_path.exists())
        self.assertTrue(self.corr_path.exists())
        self.assertTrue(self.report_path.exists())
        
        # Transform check
        df_trans = selector.transform(self.df)
        self.assertIn("Date", df_trans.columns)
        self.assertIn("Signal", df_trans.columns) # Top feature
        self.assertNotIn("Constant", df_trans.columns) # Dropped: variance
        self.assertNotIn("Duplicate", df_trans.columns) # Dropped: duplicate
        
        # Load check
        selector_loaded = FeatureSelector()
        selector_loaded.load(self.save_path)
        self.assertEqual(selector_loaded.top_k, 2)
        self.assertIn("Signal", selector_loaded.selected_features)

if __name__ == "__main__":
    unittest.main()
