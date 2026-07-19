import os
import csv
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import src.config.settings as settings
from src.utils.helpers import setup_logger

logger = setup_logger("experiment_logger")

def get_git_commit() -> str:
    """Retrieves the current Git commit hash, or returns 'unknown' if not in a Git repository."""
    try:
        # Run git rev-parse HEAD in a subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return commit
    except Exception:
        # Fallback if git is not installed or repo is not initialized
        return "unknown"

def get_dataset_version_id() -> str:
    """Retrieves the dataset version ID from dataset_version.json if it exists."""
    from src.config.settings import DATASET_VERSION_PATH
    if DATASET_VERSION_PATH.exists():
        try:
            with open(DATASET_VERSION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("Dataset Version", "unknown")
        except Exception as e:
            logger.debug(f"Could not load dataset version file: {e}")
            
    return "unknown"

def log_experiment_run(
    model_name: str,
    hyperparams: Dict[str, Any],
    train_time: float,
    inference_time: float,
    metrics: Dict[str, float],
    dataset_version: Optional[str] = None,
    feature_set_version: str = "v1.0",
    random_seed: Optional[int] = None,
    status: str = "Success",
    notes: str = ""
) -> Path:
    """
    Appends a new record of the forecast model run to the central experiments CSV.
    """
    csv_path = Path(settings.RESULTS_CSV_PATH)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    seed = random_seed if random_seed is not None else settings.FORECAST_RANDOM_SEED
    timestamp = datetime.now().isoformat()
    git_commit = get_git_commit()
    ds_version = dataset_version or get_dataset_version_id()
    
    # Extract Pinball loss keys if present in metrics (e.g. Pinball_Loss_0.1)
    pinball_losses = {k: v for k, v in metrics.items() if k.startswith("Pinball_Loss_")}
    pinball_str = json.dumps(pinball_losses) if pinball_losses else "N/A"
    
    # Compile the new row matching expected schema
    row = {
        "Timestamp": timestamp,
        "Model Name": model_name,
        "Hyperparameters": json.dumps(hyperparams),
        "Training Time": f"{train_time:.6f}",
        "Inference Time": f"{inference_time:.6f}",
        "MAE": f"{metrics.get('MAE', np.nan):.6f}" if 'MAE' in metrics else "N/A",
        "RMSE": f"{metrics.get('RMSE', np.nan):.6f}" if 'RMSE' in metrics else "N/A",
        "MAPE": f"{metrics.get('MAPE', np.nan):.6f}" if 'MAPE' in metrics else "N/A",
        "WMAPE": f"{metrics.get('WMAPE', np.nan):.6f}" if 'WMAPE' in metrics else "N/A",
        "SMAPE": f"{metrics.get('SMAPE', np.nan):.6f}" if 'SMAPE' in metrics else "N/A",
        "R2": f"{metrics.get('R2', np.nan):.6f}" if 'R2' in metrics else "N/A",
        "Pinball Loss": pinball_str,
        "Dataset Version": ds_version,
        "Feature Set Version": feature_set_version,
        "Random Seed": seed,
        "Git Commit": git_commit,
        "Status": status,
        "Notes": notes
    }
    
    headers = list(row.keys())
    file_exists = csv_path.exists()
    
    if file_exists:
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_headers = next(reader, [])
            if "WMAPE" not in existing_headers:
                logger.info("Migrating experiment results CSV to include WMAPE header column.")
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for r in rows:
                        if "WMAPE" not in r:
                            r["WMAPE"] = "N/A"
                        filtered_r = {k: v for k, v in r.items() if k in headers}
                        writer.writerow(filtered_r)
        except Exception as e:
            logger.error(f"Failed to migrate experiment results CSV: {e}")
            
    try:
        # Write to results file
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
        logger.info(f"Experiment results appended successfully to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to log experiment run: {e}")
        
    return csv_path

import numpy as np
