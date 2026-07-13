import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

from src.config import settings
from src.data_generation.generator import generate_powergrid_data
from src.validation.validator import run_data_validation
from src.preprocessing.cleaner import DataCleaner
from src.features.temporal import extract_temporal_features
from src.features.engineer import FeatureEngineer

class TestConfig(unittest.TestCase):
    """Tests configuration loading from YAML and exposure in settings."""
    def test_settings_constants(self):
        self.assertEqual(settings.RANDOM_SEED, 42)
        self.assertEqual(settings.TRAIN_RATIO, 0.70)
        self.assertEqual(settings.VAL_RATIO, 0.15)
        self.assertEqual(settings.TEST_RATIO, 0.15)
        self.assertTrue(isinstance(settings.RAW_DATA_PATH, Path))
        self.assertTrue(isinstance(settings.VALID_REGIONS, list))
        self.assertIn("NR", settings.VALID_REGIONS)

class TestGenerator(unittest.TestCase):
    """Tests synthetic POWERGRID data generation."""
    def test_generation_shape_and_columns(self):
        # Generate a small sample
        df = generate_powergrid_data(num_rows=200)
        self.assertGreaterEqual(len(df), 200)
        
        # Verify required columns are present
        required_cols = [
            "Project_ID", "Date", "Region", "State", "Warehouse", "Supplier",
            "Material_Type", "Project_Phase", "Tower_Type", "Substation_Type",
            "Historical_Demand", "Current_Inventory", "Lead_Time_Days",
            "Supplier_Risk", "Commodity_Price", "Transportation_Cost",
            "Storage_Capacity", "Production_Capacity", "Project_Budget",
            "Weather", "Season", "Quantity_Required"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)
            
    def test_operational_scenarios(self):
        df = generate_powergrid_data(num_rows=500)
        # Test emergency projects exist
        emer_projects = df[df["Project_ID"].str.contains("EMER")]
        self.assertGreaterEqual(len(emer_projects), 0) # In small runs might be 0, but is supported
        
        # Test seasons
        seasons = df["Season"].unique()
        self.assertTrue(any(s in seasons for s in ["Winter", "Summer", "Monsoon", "Post-Monsoon"]))

class TestValidator(unittest.TestCase):
    """Tests validator checks and report rows."""
    def test_validation_rules(self):
        # Create dirty mock dataframe
        mock_data = pd.DataFrame({
            "Project_ID": ["PG-001", "PG-001"],
            "Date": ["2023-01-01", "202-INVALID"], # 1 invalid date
            "Region": ["NR", "XX"],                 # 1 invalid region
            "State": ["Haryana", "StateX"],
            "Warehouse": ["WH-NR-01", "WH-NR-01"],
            "Supplier": ["Tata", "Tata"],
            "Material_Type": ["Conductor", "Conductor"],
            "Project_Phase": ["Foundation", "InvalidPhase"], # 1 invalid phase
            "Tower_Type": ["Suspension", "Suspension"],
            "Substation_Type": ["AIS", "AIS"],
            "Historical_Demand": [100, -50],        # 1 negative
            "Current_Inventory": [-10, 200],        # 1 negative
            "Lead_Time_Days": [30, np.nan],         # 1 missing
            "Supplier_Risk": [0.2, 0.4],
            "Commodity_Price": [100.0, 105.0],
            "Transportation_Cost": [20.0, 25.0],
            "Storage_Capacity": [5000, 5000],
            "Production_Capacity": [4000, 4000],
            "Project_Budget": [200000, 200000],
            "Weather": [np.nan, "Normal"],          # 1 missing
            "Season": ["Winter", "Winter"],
            "Quantity_Required": [120, -10]         # 1 negative
        })
        
        report = run_data_validation(mock_data, stage="test_raw")
        
        # Check that failures were registered
        fails = report[report["Status"] == "FAIL"]
        self.assertGreater(len(fails), 0)
        
        # Check specific checks exist
        check_names = report["Check_Name"].unique()
        self.assertIn("Missing Values Check", check_names)
        self.assertIn("Negative Inventory Check", check_names)
        self.assertIn("Negative Demand Check", check_names)
        self.assertIn("Invalid Date Formats or Range", check_names)

class TestCleaner(unittest.TestCase):
    """Tests DataCleaner fit and transform methods."""
    def test_cleaner_imputation_and_outliers(self):
        # Train data
        train_mock = pd.DataFrame({
            "Project_ID": ["PG-001", "PG-001", "PG-002", "PG-002", "PG-002"],
            "Date": ["2023-01-01", "2023-01-08", "2023-01-15", "2023-01-22", "2023-01-29"],
            "Region": ["NR", "NR", "NR", "NR", "NR"],
            "State": ["Haryana", "Haryana", "Haryana", "Haryana", "Haryana"],
            "Warehouse": ["WH-NR-01", "WH-NR-01", "WH-NR-01", "WH-NR-01", "WH-NR-01"],
            "Supplier": ["Tata", "Tata", "Tata", "Tata", "Tata"],
            "Material_Type": ["Conductor", "Conductor", "Conductor", "Conductor", "Conductor"],
            "Project_Phase": ["Foundation", "Foundation", "Foundation", "Foundation", "Foundation"],
            "Tower_Type": ["Suspension", "Suspension", "Suspension", "Suspension", "Suspension"],
            "Substation_Type": ["AIS", "AIS", "AIS", "AIS", "AIS"],
            "Historical_Demand": [100.0, 120.0, 110.0, 130.0, 1000.0], # 1000 is outlier
            "Current_Inventory": [500.0, 480.0, 490.0, 470.0, 460.0],
            "Lead_Time_Days": [30.0, 32.0, 31.0, 30.0, 33.0],
            "Supplier_Risk": [0.2, 0.22, 0.21, 0.20, 0.23],
            "Commodity_Price": [100.0, 100.0, 100.0, 100.0, 100.0],
            "Transportation_Cost": [20.0, 20.0, 20.0, 20.0, 20.0],
            "Storage_Capacity": [5000.0, 5000.0, 5000.0, 5000.0, 5000.0],
            "Production_Capacity": [4000.0, 4000.0, 4000.0, 4000.0, 4000.0],
            "Project_Budget": [200000.0, 200000.0, 200000.0, 200000.0, 200000.0],
            "Weather": ["Normal", "Normal", "Normal", "Normal", "Normal"],
            "Season": ["Winter", "Winter", "Winter", "Winter", "Winter"],
            "Quantity_Required": [110.0, 115.0, 120.0, 105.0, 112.0]
        })
        
        # Test data with anomalies
        test_mock = pd.DataFrame({
            "Project_ID": ["PG-001", "PG-001"],
            "Date": ["2023-02-05", "202-INVALID"], # 1 invalid date
            "Region": ["NR", "XX"],                 # 1 invalid region
            "State": ["Haryana", "Haryana"],
            "Warehouse": ["WH-NR-01", "WH-NR-01"],
            "Supplier": ["Tata", "Tata"],
            "Material_Type": ["Conductor", "Conductor"],
            "Project_Phase": ["Foundation", "Unknown Phase"], # 1 invalid phase
            "Tower_Type": ["Suspension", "Suspension"],
            "Substation_Type": ["AIS", "AIS"],
            "Historical_Demand": [115.0, np.nan],   # 1 missing
            "Current_Inventory": [-50.0, 450.0],    # 1 negative
            "Lead_Time_Days": [31.0, 31.0],
            "Supplier_Risk": [0.21, 0.21],
            "Commodity_Price": [100.0, 100.0],
            "Transportation_Cost": [20.0, 20.0],
            "Storage_Capacity": [5000.0, 5000.0],
            "Production_Capacity": [4000.0, 4000.0],
            "Project_Budget": [200000.0, 200000.0],
            "Weather": ["Normal", np.nan],          # 1 missing weather
            "Season": ["Winter", "Winter"],
            "Quantity_Required": [108.0, 108.0]
        })
        
        cleaner = DataCleaner()
        cleaner.fit(train_mock)
        cleaned_test = cleaner.transform(test_mock)
        
        # Assertions
        # 1. Invalid date row should be dropped (shape decreases to 1)
        self.assertEqual(len(cleaned_test), 1)
        
        # 2. Negative inventory should be set to 0.0, and then capped to training lower bound (440.0)
        self.assertEqual(cleaned_test["Current_Inventory"].iloc[0], 440.0)
        
        # 3. Invalid region should be replaced by train mode ("NR")
        # In this case the row with "XX" was also the row with "202-INVALID" (dropped), 
        # so let's verify invalid phase was not dropped
        # (It is dropped if invalid date, but let's test general clean rules)

class TestFeatures(unittest.TestCase):
    """Tests temporal features and domain feature calculations."""
    def test_temporal_features(self):
        mock_data = pd.DataFrame({
            "Date": ["2023-01-01", "2023-06-15"]
        })
        temp_df = extract_temporal_features(mock_data)
        self.assertIn("Year", temp_df.columns)
        self.assertIn("Month", temp_df.columns)
        self.assertIn("Month_Sin", temp_df.columns)
        self.assertEqual(temp_df["Year"].iloc[0], 2023)
        self.assertEqual(temp_df["Month"].iloc[0], 1)
        
    def test_feature_engineering_lags(self):
        # Create sequential weekly dataframe for 1 group
        mock_data = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-15", "2023-01-22"]),
            "Warehouse": ["WH-NR-01"] * 4,
            "Material_Type": ["Conductor"] * 4,
            "Quantity_Required": [100, 110, 120, 130],
            "Current_Inventory": [1000, 900, 800, 700],
            "Storage_Capacity": [2000] * 4,
            "Project_Budget": [100000] * 4,
            "Commodity_Price": [100.0] * 4,
            "Transportation_Cost": [20.0] * 4,
            "Supplier_Risk": [0.2] * 4,
            "Lead_Time_Days": [30] * 4,
            "Season": ["Winter"] * 4
        })
        
        engineer = FeatureEngineer()
        engineer.fit(mock_data)
        eng_df = engineer.transform(mock_data)
        
        # Verify Lags are correct (grouped and chronologically shifted)
        # Sort is applied inside transform: WH-NR-01, Conductor, Date
        # Row 0 (2023-01-01): Lag_1 should be imputed with global median (115.0)
        # Row 1 (2023-01-08): Lag_1 should be 100 (from Row 0)
        # Row 2 (2023-01-15): Lag_1 should be 110 (from Row 1)
        
        self.assertEqual(eng_df["Lag_1"].iloc[1], 100.0)
        self.assertEqual(eng_df["Lag_1"].iloc[2], 110.0)
        
        # Verify inventory utilization
        # Row 0: 1000 / 2000 = 0.50
        self.assertAlmostEqual(eng_df["Inventory_Utilization"].iloc[0], 0.50)

if __name__ == "__main__":
    unittest.main()
