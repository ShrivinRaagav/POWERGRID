import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
import os

from src.time_series.utils import prepare_time_series
from src.time_series.dwt import DWTTransformer
from src.time_series.emd import EMDProcessor
from src.time_series.decomposition import TimeSeriesFeatureExtractor
from src.time_series.visualization import plot_group_decompositions

class TestTimeSeriesUtils(unittest.TestCase):
    """Tests for the time-series preparation and timestamp filling utility functions."""
    def setUp(self):
        # Create a mock dataset with missing weeks for 2 groups
        self.df = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-08", "2023-01-22",  # group A has missing 2023-01-15
                     "2023-01-01", "2023-01-15", "2023-01-22"], # group B has missing 2023-01-08
            "Material_Type": ["Conductor", "Conductor", "Conductor", "Insulator", "Insulator", "Insulator"],
            "Region": ["NR", "NR", "NR", "ER", "ER", "ER"],
            "Quantity_Required": [100.0, 110.0, 130.0, 200.0, 220.0, 240.0]
        })

    def test_prepare_time_series_grouping_and_reindex(self):
        # Test zero-fill method
        df_prepared = prepare_time_series(
            self.df,
            group_cols=["Material_Type", "Region"],
            date_col="Date",
            target_col="Quantity_Required",
            freq="W-SUN",
            fill_method="zero"
        )
        
        # Expected dates are 4 Sundays: 2023-01-01, 2023-01-08, 2023-01-15, 2023-01-22
        # Since we have 2 groups, we expect 2 * 4 = 8 rows in total
        self.assertEqual(len(df_prepared), 8)
        
        # Group A (Conductor, NR) at 2023-01-15 should be zero filled
        cond_nr = df_prepared[(df_prepared["Material_Type"] == "Conductor") & (df_prepared["Region"] == "NR")]
        cond_nr_sorted = cond_nr.sort_values("Date")
        self.assertEqual(cond_nr_sorted["Quantity_Required"].iloc[2], 0.0) # 2023-01-15 index is 2
        
        # Group B (Insulator, ER) at 2023-01-08 should be zero filled
        ins_er = df_prepared[(df_prepared["Material_Type"] == "Insulator") & (df_prepared["Region"] == "ER")]
        ins_er_sorted = ins_er.sort_values("Date")
        self.assertEqual(ins_er_sorted["Quantity_Required"].iloc[1], 0.0) # 2023-01-08 index is 1

    def test_prepare_time_series_ffill(self):
        df_prepared = prepare_time_series(
            self.df,
            group_cols=["Material_Type", "Region"],
            date_col="Date",
            target_col="Quantity_Required",
            freq="W-SUN",
            fill_method="ffill"
        )
        # Group A at 2023-01-15 should be filled with 110.0 (from 2023-01-08)
        cond_nr = df_prepared[(df_prepared["Material_Type"] == "Conductor") & (df_prepared["Region"] == "NR")]
        cond_nr_sorted = cond_nr.sort_values("Date")
        self.assertEqual(cond_nr_sorted["Quantity_Required"].iloc[2], 110.0)

    def test_prepare_time_series_interpolate(self):
        df_prepared = prepare_time_series(
            self.df,
            group_cols=["Material_Type", "Region"],
            date_col="Date",
            target_col="Quantity_Required",
            freq="W-SUN",
            fill_method="interpolate"
        )
        # Group A at 2023-01-15 should be filled with linear interpolation: (110.0 + 130.0) / 2 = 120.0
        cond_nr = df_prepared[(df_prepared["Material_Type"] == "Conductor") & (df_prepared["Region"] == "NR")]
        cond_nr_sorted = cond_nr.sort_values("Date")
        self.assertEqual(cond_nr_sorted["Quantity_Required"].iloc[2], 120.0)


class TestDWT(unittest.TestCase):
    """Tests for the Discrete Wavelet Transform (DWT) module."""
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.save_path = self.temp_dir / "dwt.joblib"
        # Create a basic sine wave signal
        t = np.linspace(0, 1, 64)
        self.signal = np.sin(2 * np.pi * 5 * t) + 10.0
        
        # Build a small dummy dataframe
        dates = pd.date_range("2023-01-01", periods=64, freq="W")
        self.df = pd.DataFrame({
            "Date": dates,
            "Quantity_Required": self.signal,
            "Material_Type": ["Conductor"] * 64,
            "Region": ["NR"] * 64
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_decomposition_keys_and_shapes(self):
        dwt = DWTTransformer(wavelet="db4", level=2, save_path=self.save_path)
        res = dwt.decompose_signal(self.signal)
        
        self.assertIn("cA", res)
        self.assertIn("cD_1", res)
        self.assertIn("cD_2", res)
        self.assertEqual(len(res["cA"]), len(res["cD_2"]))
        
    def test_fit_transform_and_serialization(self):
        dwt = DWTTransformer(wavelet="db4", level=2, save_path=self.save_path)
        dwt.fit_transform(self.df, group_cols=["Material_Type", "Region"])
        
        self.assertTrue(dwt.is_fit)
        self.assertTrue(self.save_path.exists())
        
        # Load and verify
        dwt_loaded = DWTTransformer()
        dwt_loaded.load(self.save_path)
        self.assertEqual(dwt_loaded.wavelet, "db4")
        self.assertEqual(dwt_loaded.level, 2)
        self.assertIn(("Conductor", "NR"), dwt_loaded.decomposition_results)


class TestEMD(unittest.TestCase):
    """Tests for the Empirical Mode Decomposition (EMD) module."""
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.save_path = self.temp_dir / "emd.joblib"
        # Sine wave with high-freq noise and trend
        t = np.linspace(0, 2, 100)
        self.signal = np.sin(2 * np.pi * 2 * t) + 0.2 * np.sin(2 * np.pi * 10 * t) + 2.0 * t
        
        # Build small dummy dataframe
        dates = pd.date_range("2023-01-01", periods=100, freq="W")
        self.df = pd.DataFrame({
            "Date": dates,
            "Quantity_Required": self.signal,
            "Material_Type": ["Conductor"] * 100,
            "Region": ["NR"] * 100
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_emd_imfs_reconstruction(self):
        emd = EMDProcessor(spline_kind="cubic", max_imfs=3, save_path=self.save_path)
        res = emd.decompose_signal(self.signal)
        
        self.assertIn("imfs", res)
        self.assertIn("residual", res)
        
        imfs = res["imfs"]
        residual = res["residual"]
        
        # Reconstruct original signal: signal = sum(imfs) + residual
        reconstructed = np.sum(imfs, axis=0) + residual
        np.testing.assert_allclose(self.signal, reconstructed, rtol=1e-5, atol=1e-5)

    def test_emd_fit_transform_and_serialization(self):
        emd = EMDProcessor(spline_kind="cubic", max_imf=3, save_path=self.save_path)
        emd.fit_transform(self.df, group_cols=["Material_Type", "Region"])
        
        self.assertTrue(emd.is_fit)
        self.assertTrue(self.save_path.exists())
        
        # Load and verify
        emd_loaded = EMDProcessor()
        emd_loaded.load(self.save_path)
        self.assertEqual(emd_loaded.spline_kind, "cubic")
        self.assertEqual(emd_loaded.max_imf, 3)
        self.assertIn(("Conductor", "NR"), emd_loaded.decomposition_results)


class TestDecompositionExtractor(unittest.TestCase):
    """Tests for the combined TimeSeriesFeatureExtractor pipeline class."""
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create temporary file paths
        self.dwt_path = self.temp_dir / "dwt_transform.joblib"
        self.emd_path = self.temp_dir / "emd_processor.joblib"
        self.extractor_path = self.temp_dir / "feature_extractor.joblib"
        self.summary_path = self.temp_dir / "decomposition_summary.csv"
        self.plots_dir = self.temp_dir / "plots"
        
        # Sine wave with trend
        t = np.linspace(0, 3, 150)
        self.signal = np.sin(2 * np.pi * 3 * t) + 1.5 * t
        
        dates = pd.date_range("2023-01-01", periods=150, freq="W")
        self.df = pd.DataFrame({
            "Date": dates,
            "Quantity_Required": self.signal,
            "Material_Type": ["Conductor"] * 150,
            "Region": ["NR"] * 150
        })

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_extractor_pipeline(self):
        # Override settings paths temporarily via constructor injection, or by configuring extractor
        extractor = TimeSeriesFeatureExtractor(
            group_cols=["Material_Type", "Region"],
            date_col="Date",
            target_col="Quantity_Required",
            save_path=self.extractor_path
        )
        
        # Inject temp paths to sub-processors to prevent writing to real artifacts directory during tests
        extractor.dwt_transformer.save_path = self.dwt_path
        extractor.emd_processor.save_path = self.emd_path
        
        # Mock SUMMARY_PATH in module
        import src.time_series.decomposition as decomp_module
        original_summary_path = decomp_module.TS_SUMMARY_PATH
        decomp_module.TS_SUMMARY_PATH = self.summary_path
        
        try:
            df_enriched, df_summary = extractor.fit_transform(self.df)
            
            # Assertions
            self.assertTrue(extractor.is_fit)
            self.assertTrue(self.extractor_path.exists())
            self.assertTrue(self.summary_path.exists())
            
            # Verify summary contains statistical properties
            self.assertIn("Signal_Length", df_summary.columns)
            self.assertIn("Trend_Strength", df_summary.columns)
            self.assertIn("Seasonality_Strength", df_summary.columns)
            self.assertIn("Dominant_Frequency", df_summary.columns)
            self.assertEqual(df_summary["Signal_Length"].iloc[0], 150)
            
            # Verify enriched dataframe contains signals
            self.assertIn("Classical_Trend", df_enriched.columns)
            self.assertIn("Classical_Seasonal", df_enriched.columns)
            self.assertIn("Classical_Residual", df_enriched.columns)
            self.assertIn("EMD_Residual", df_enriched.columns)
            self.assertIn("EMD_IMF_1", df_enriched.columns)
            self.assertIn("Signal_Energy", df_enriched.columns) # Static feature
            
            # Test plotting function on a single group
            key = ("Conductor", "NR")
            dwt_res = extractor.dwt_transformer.decomposition_results[key]
            emd_res = extractor.emd_processor.decomposition_results[key]
            
            plot_group_decompositions(
                group_key=key,
                df_group=df_enriched,
                dwt_coeffs=dwt_res,
                emd_results=emd_res,
                date_col="Date",
                target_col="Quantity_Required",
                output_dir=self.plots_dir,
                dpi=100
            )
            
            self.assertTrue((self.plots_dir / "Conductor_NR_classical_decomposition.png").exists())
            self.assertTrue((self.plots_dir / "Conductor_NR_dwt_decomposition.png").exists())
            self.assertTrue((self.plots_dir / "Conductor_NR_emd_decomposition.png").exists())
            
        finally:
            # Restore module variables
            decomp_module.TS_SUMMARY_PATH = original_summary_path

if __name__ == "__main__":
    unittest.main()
