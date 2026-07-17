import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Tuple, Any
import pandas as pd

from src.config.settings import EXPERIMENTS_DIR
from src.utils.helpers import setup_logger

logger = setup_logger("experiment_manager")

class ExperimentManager:
    """
    Handles file organization and subdirectory creation for individual forecasting model runs.
    Automatically generates run IDs and encapsulates output directories for results, 
    plots, checkpoints, and execution logs.
    """
    def __init__(self, model_name: str, run_id: Optional[str] = None):
        self.model_name = model_name.strip().lower()
        
        if run_id:
            self.run_id = run_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"EXP-{self.model_name}-{timestamp}"
            
        # Assign run directories
        self.run_dir = EXPERIMENTS_DIR
        self.checkpoints_dir = EXPERIMENTS_DIR / "checkpoints" / self.run_id
        self.results_dir = EXPERIMENTS_DIR / "results" / self.run_id
        self.plots_dir = EXPERIMENTS_DIR / "plots" / self.run_id
        self.logs_dir = EXPERIMENTS_DIR / "logs" / self.run_id
        
        self.create_directories()

    def create_directories(self) -> None:
        """Creates the run directories on the filesystem."""
        for path in [self.checkpoints_dir, self.results_dir, self.plots_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized directories for experiment run: {self.run_id}")

    def save_plot(self, fig, filename: str, dpi: int = 150) -> Path:
        """Saves a matplotlib figure directly to the run's plots directory."""
        import matplotlib.pyplot as plt
        save_path = self.plots_dir / filename
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved experiment plot: {save_path}")
        return save_path

    def save_predictions(self, df_preds: pd.DataFrame, filename: str = "predictions.csv") -> Path:
        """Saves a predictions DataFrame directly to the run's results directory."""
        save_path = self.results_dir / filename
        df_preds.to_csv(save_path, index=False)
        logger.info(f"Saved experiment predictions: {save_path}")
        return save_path

    def save_model_checkpoint(self, model: Any, filename: str = "model.joblib") -> Path:
        """Serializes and saves a model checkpoint directly to the run's checkpoints directory."""
        save_path = self.checkpoints_dir / filename
        model.save(str(save_path))
        logger.info(f"Saved model checkpoint: {save_path}")
        return save_path

    def get_run_log_path(self, filename: str = "run.log") -> Path:
        """Returns the path for saving execution logs for this specific run."""
        return self.logs_dir / filename

import typing
Any = typing.Any
