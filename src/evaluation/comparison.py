import pandas as pd
from typing import Dict, Any, List, Optional
from src.utils.helpers import setup_logger

logger = setup_logger("comparison_registry")

class ModelComparisonRegistry:
    """
    Utility class to register, compare, and rank different forecasting models' 
    evaluation metrics (e.g. comparing LSTM, RandomForest, SVR, etc.).
    """
    def __init__(self):
        self.registry: Dict[str, Dict[str, float]] = {}

    def register_model(self, model_name: str, metrics: Dict[str, float]):
        """Registers a model's evaluation metrics."""
        self.registry[model_name] = metrics.copy()
        logger.info(f"Registered model '{model_name}' metrics: {metrics}")

    def get_comparison_table(self) -> pd.DataFrame:
        """Compiles registered models' metrics into a comparative DataFrame."""
        if not self.registry:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(self.registry, orient="index")
        df.index.name = "Model_Name"
        return df.reset_index()

    def get_best_model(self, rank_by_metric: str = "MAE", ascending: bool = True) -> Optional[str]:
        """
        Identifies the best-performing model based on the selected metric.
        
        Parameters:
        rank_by_metric (str): The metric to evaluate (default "MAE").
        ascending (bool): True if lower metric is better (MAE, RMSE, MAPE), False otherwise (R2).
        
        Returns:
        str: Name of the best model.
        """
        df = self.get_comparison_table()
        if df.empty:
            logger.warning("Comparison registry is empty.")
            return None
            
        if rank_by_metric not in df.columns:
            logger.warning(f"Metric '{rank_by_metric}' not found in registry columns: {list(df.columns)}")
            return None
            
        best_row = df.sort_values(by=rank_by_metric, ascending=ascending).iloc[0]
        best_model = best_row["Model_Name"]
        logger.info(f"Best model ranked by {rank_by_metric} is '{best_model}' (Value: {best_row[rank_by_metric]:.6f})")
        return best_model

    def save_comparison_csv(self, filepath: str):
        """Saves comparison table to a CSV file."""
        df = self.get_comparison_table()
        if df.empty:
            logger.warning("No data to save in comparison registry.")
            return
            
        df.to_csv(filepath, index=False)
        logger.info(f"Saved model comparison table to {filepath}")
