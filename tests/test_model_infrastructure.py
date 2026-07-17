import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
import json
from pathlib import Path

from src.models.base_model import BaseForecastModel
from src.models.registry import register_model, get_model_class, list_registered_models
from src.experiments.experiment_manager import ExperimentManager
from src.experiments.experiment_logger import log_experiment_run, get_git_commit

class DummyForecastModel(BaseForecastModel):
    """Fulfills interface for testing registry class matching."""
    def __init__(self, p1=1):
        self.p1 = p1
        self.fitted = False
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.fitted = True
    def predict(self, X):
        return np.ones(len(X))
    def evaluate(self, X, y):
        return {"MAE": 0.0}
    def save(self, filepath):
        pass
    def load(self, filepath):
        pass
    def get_model_name(self) -> str:
        return "dummy_test"
    def get_hyperparameters(self):
        return {"p1": self.p1}

class TestModelRegistry(unittest.TestCase):
    def test_registry_registration_and_retrieval(self):
        # Decorate and register
        registered_cls = register_model("test_registry_mock")(DummyForecastModel)
        
        # Verify lists
        models_list = list_registered_models()
        self.assertIn("test_registry_mock", models_list)
        
        # Retrieve class
        retrieved_cls = get_model_class("test_registry_mock")
        self.assertEqual(retrieved_cls, DummyForecastModel)
        
        # Test case insensitivity
        self.assertEqual(get_model_class("TEST_REGISTRY_MOCK"), DummyForecastModel)

    def test_invalid_class_registration(self):
        # Class that does not inherit from BaseForecastModel
        class NonConformingModel:
            pass
            
        with self.assertRaises(TypeError):
            register_model("non_conforming")(NonConformingModel)

    def test_missing_abstract_methods_instantiation(self):
        # Create class dynamically to avoid static type checker warnings
        missing_cls = type("MissingMethodsModel", (BaseForecastModel,), {})
        with self.assertRaises(TypeError):
            missing_cls()


class TestExperimentTracking(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_csv_path = None
        self.original_experiments_dir = None
        
        # Mock module variables
        import src.config.settings as settings
        self.original_csv_path = settings.RESULTS_CSV_PATH
        self.original_experiments_dir = settings.EXPERIMENTS_DIR
        
        settings.RESULTS_CSV_PATH = self.temp_dir / "test_results.csv"
        settings.EXPERIMENTS_DIR = self.temp_dir / "experiments"

    def tearDown(self):
        # Restore module variables
        import src.config.settings as settings
        settings.RESULTS_CSV_PATH = self.original_csv_path
        settings.EXPERIMENTS_DIR = self.original_experiments_dir
        shutil.rmtree(self.temp_dir)

    def test_experiment_manager_folders_creation(self):
        manager = ExperimentManager(model_name="test_rf")
        
        self.assertTrue(manager.checkpoints_dir.exists())
        self.assertTrue(manager.results_dir.exists())
        self.assertTrue(manager.plots_dir.exists())
        self.assertTrue(manager.logs_dir.exists())
        
        # Test save predictions
        df_pred = pd.DataFrame({"Actual": [1, 2], "Predicted": [1.1, 2.1]})
        pred_path = manager.save_predictions(df_pred, "test_preds.csv")
        self.assertTrue(pred_path.exists())
        self.assertEqual(pd.read_csv(pred_path).shape, (2, 2))

    def test_experiment_logger_writes(self):
        metrics = {"MAE": 0.05, "RMSE": 0.1, "R2": 0.95}
        hyperparams = {"n_estimators": 50}
        
        csv_path = log_experiment_run(
            model_name="mock_test",
            hyperparams=hyperparams,
            train_time=1.5,
            inference_time=0.02,
            metrics=metrics,
            dataset_version="DS-test",
            random_seed=42
        )
        
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["Model Name"], "mock_test")
        self.assertEqual(df.iloc[0]["MAE"], 0.05)
        self.assertEqual(json.loads(df.iloc[0]["Hyperparameters"])["n_estimators"], 50)
        self.assertEqual(df.iloc[0]["Dataset Version"], "DS-test")
        
        # Check git commit
        git_commit = get_git_commit()
        self.assertEqual(df.iloc[0]["Git Commit"], git_commit)

if __name__ == "__main__":
    unittest.main()
