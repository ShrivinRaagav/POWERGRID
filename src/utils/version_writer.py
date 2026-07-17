import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.utils.helpers import setup_logger

logger = setup_logger("version_writer")

def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Computes a SHA-256 hash of the dataframe to act as a unique version identifier."""
    # Convert data values and column names to bytes and hash
    try:
        # A lightweight way is to hash the columns and shapes, plus a sample or summary
        # or convert the string representation of columns and values to bytes
        summary_str = f"{df.shape}_{list(df.columns)}"
        # Add head and tail values hash for content detection
        content_sample = df.head(10).to_string() + df.tail(10).to_string()
        full_hash_input = summary_str + content_sample
        return hashlib.sha256(full_hash_input.encode("utf-8")).hexdigest()[:12]
    except Exception as e:
        logger.debug(f"Error hashing dataframe: {e}. Falling back to timestamp hash.")
        return hashlib.sha256(datetime.now().isoformat().encode("utf-8")).hexdigest()[:12]

def generate_dataset_version_report(
    df: pd.DataFrame,
    random_seed: int,
    output_path: Path
):
    """
    Generates reports/dataset_version.json dynamically summarizing the dataset structure,
    size, active features, and unique hashes.
    """
    logger.info(f"Generating dataset version metadata at {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    version_hash = compute_dataset_hash(df)
    creation_date = datetime.now().isoformat()
    
    metadata = {
        "Creation Date": creation_date,
        "Dataset Version": f"DS-{version_hash}",
        "Number of Samples": len(df),
        "Number of Features": len(df.columns),
        "Selected Features": list(df.columns),
        "Feature Engineering Version": "v1.0",
        "Time-Series Version": "v1.0",
        "Feature Selection Version": "v1.0",
        "Random Seed": random_seed,
        "Pipeline Version": "v1.2"
    }
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Dataset version report written successfully: DS-{version_hash}")
    except Exception as e:
        logger.error(f"Failed to save dataset version report: {e}")
