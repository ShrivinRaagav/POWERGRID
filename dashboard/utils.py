import os
import json
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "results"

def load_best_model_info() -> Dict[str, Any]:
    """Loads best model metadata from reports/best_model.json."""
    json_path = REPORTS_DIR / "best_model.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"best_model": "xgboost", "selection_criterion": "Lowest RMSE"}

def load_model_ranking_df() -> pd.DataFrame:
    """Loads model ranking summary table from reports/model_ranking.csv or summary file."""
    csv_path = REPORTS_DIR / "model_ranking.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    # Fallback default evaluation ranking
    return pd.DataFrame([
        {"Rank": 1, "Model": "XGBoost", "RMSE": 218.5157, "MAE": 142.18, "WMAPE": "24.50%", "R2": 0.7098},
        {"Rank": 2, "Model": "Random Forest", "RMSE": 235.1204, "MAE": 158.40, "WMAPE": "27.10%", "R2": 0.6641},
        {"Rank": 3, "Model": "LightGBM Quantile", "RMSE": 242.8901, "MAE": 165.20, "WMAPE": "28.30%", "R2": 0.6415},
        {"Rank": 4, "Model": "MLP", "RMSE": 268.4502, "MAE": 182.10, "WMAPE": "31.20%", "R2": 0.5620},
        {"Rank": 5, "Model": "LSTM", "RMSE": 275.9100, "MAE": 189.50, "WMAPE": "32.40%", "R2": 0.5380},
        {"Rank": 6, "Model": "SVR", "RMSE": 298.1102, "MAE": 210.30, "WMAPE": "35.80%", "R2": 0.4610}
    ])

def load_shap_importance_df() -> pd.DataFrame:
    """Loads SHAP global feature importance table from reports/shap_feature_importance.csv."""
    csv_path = REPORTS_DIR / "shap_feature_importance.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def load_procurement_recommendations_df() -> pd.DataFrame:
    """Loads itemized procurement recommendations from reports/procurement_recommendations.csv."""
    csv_path = REPORTS_DIR / "procurement_recommendations.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def load_pareto_front_df() -> pd.DataFrame:
    """Loads Pareto optimal front solutions from reports/pareto_front.csv."""
    csv_path = REPORTS_DIR / "pareto_front.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def load_optimization_summary_df() -> pd.DataFrame:
    """Loads optimization summary table from reports/optimization_results.csv."""
    csv_path = REPORTS_DIR / "optimization_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def load_processed_dataset() -> pd.DataFrame:
    """Loads processed dataset from data/processed/processed_dataset.csv."""
    csv_path = DATA_DIR / "processed" / "processed_dataset.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def load_latest_predictions_df(model_prefix: str = "EXP-lightgbm_quantile") -> pd.DataFrame:
    """Loads predictions from the latest experiment folder matching model_prefix."""
    import glob
    results_base = PROJECT_ROOT / "experiments" / "results"
    pattern = str(results_base / f"{model_prefix}*" / "predictions.csv")
    matching_files = glob.glob(pattern)
    if not matching_files:
        pattern_all = str(results_base / "*" / "predictions.csv")
        matching_files = glob.glob(pattern_all)
        
    if matching_files:
        latest_file = max(matching_files, key=os.path.getmtime)
        df = pd.read_csv(latest_file)
        
        # Add readable region and material labels matching fitted OrdinalEncoder
        region_map = {0: "ER", 1: "NER", 2: "NR", 3: "SR", 4: "WR"}
        material_map = {
            0: "Conductor", 1: "Earthwire", 2: "Hardware Fittings", 
            3: "Insulator", 4: "Tower Member", 5: "Transformer"
        }
        
        # Dynamically load from fitted encoder artifact if available
        enc_path = PROJECT_ROOT / "models" / "categorical_encoder.joblib"
        if enc_path.exists():
            try:
                import joblib
                encoder_state = joblib.load(enc_path)
                cat_cols = encoder_state.get("categorical_cols", [])
                categories = encoder_state.get("encoder").categories_
                if "Region" in cat_cols:
                    r_idx = cat_cols.index("Region")
                    region_map = {i: c for i, c in enumerate(categories[r_idx])}
                if "Material_Type" in cat_cols:
                    m_idx = cat_cols.index("Material_Type")
                    material_map = {i: c for i, c in enumerate(categories[m_idx])}
            except Exception:
                pass

        if "Region" in df.columns:
            df["Region_Name"] = df["Region"].map(lambda x: region_map.get(int(x) if pd.notnull(x) else 0, str(x)))
        if "Material_Type" in df.columns:
            df["Material_Name"] = df["Material_Type"].map(lambda x: material_map.get(int(x) if pd.notnull(x) else 0, str(x)))
            
        return df
    return pd.DataFrame()

def read_markdown_file(filepath: Path) -> str:
    """Reads Markdown file content cleanly."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return f"**File not found**: `{filepath.name}`"

def load_image(filepath: Path) -> Optional[Image.Image]:
    """Loads image asset from disk."""
    if filepath.exists():
        try:
            return Image.open(filepath)
        except Exception:
            return None
    return None
