import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union

from src.config.settings import (
    PROCESSED_DATASET_PATH, EXPERIMENTS_DIR, REPORTS_DIR, DATE_COL, TARGET_COL
)
from src.utils.helpers import setup_logger

logger = setup_logger("shap_explainer")

def get_best_model_info(reports_dir: Path = REPORTS_DIR) -> Dict[str, Any]:
    """
    Loads best model metadata from reports/best_model.json.
    """
    json_path = reports_dir / "best_model.json"
    if not json_path.exists():
        raise FileNotFoundError(f"best_model.json not found at {json_path}. Run model training first.")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(f"Loaded best model info: {meta.get('best_model', 'unknown')}")
    return meta

def load_best_model_checkpoint(best_model_name: str, experiments_dir: Path = EXPERIMENTS_DIR) -> Tuple[Any, Path]:
    """
    Locates and loads the latest trained model checkpoint for the given model name.
    """
    checkpoints_root = experiments_dir / "checkpoints"
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints directory not found at {checkpoints_root}")

    matching_dirs = list(checkpoints_root.glob(f"EXP-{best_model_name.lower()}-*"))
    if not matching_dirs:
        raise FileNotFoundError(f"No checkpoint directories found for model '{best_model_name}' in {checkpoints_root}")

    # Sort by modification time to get latest run
    matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_dir = matching_dirs[0]

    model_file = latest_dir / "model.joblib"
    if not model_file.exists():
        # Fallback to any joblib file in directory
        joblibs = list(latest_dir.glob("*.joblib"))
        if not joblibs:
            raise FileNotFoundError(f"No .joblib checkpoint found in {latest_dir}")
        model_file = joblibs[0]

    model_obj = joblib.load(model_file)
    logger.info(f"Successfully loaded trained model '{best_model_name}' from {model_file}")
    return model_obj, model_file

def load_processed_test_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    Loads processed_dataset.csv, chronologically splits (70% train, 15% val, 15% test),
    and returns (X_test, y_test, X_train, feature_names).
    """
    if not PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATASET_PATH}")

    df = pd.read_csv(PROCESSED_DATASET_PATH)

    # Sort chronologically
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.sort_values(by=DATE_COL).reset_index(drop=True)

    unique_dates = df[DATE_COL].unique() if DATE_COL in df.columns else np.arange(len(df))
    n_dates = len(unique_dates)

    n_train = int(n_dates * 0.70)
    n_val = int(n_dates * 0.15)

    if DATE_COL in df.columns:
        train_dates = unique_dates[:n_train]
        test_dates = unique_dates[n_train + n_val:]
        train_df = df[df[DATE_COL].isin(train_dates)]
        test_df = df[df[DATE_COL].isin(test_dates)]
    else:
        train_df = df.iloc[:int(len(df) * 0.7)]
        test_df = df.iloc[int(len(df) * 0.85):]

    id_cols = [c for c in ["Date", "Project_ID", "Region", "Material_Type"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in id_cols and c != TARGET_COL]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy() if TARGET_COL in test_df.columns else pd.Series()

    logger.info(f"Loaded test dataset: X_test shape={X_test.shape}, features={len(feature_cols)}")
    return X_test, y_test, X_train, feature_cols

class SHAPExplainerManager:
    """
    Manages automatic SHAP explainer selection, computation, and caching for the best model.
    """
    def __init__(self, reports_dir: Path = REPORTS_DIR, experiments_dir: Path = EXPERIMENTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.experiments_dir = Path(experiments_dir)

        self.best_meta = get_best_model_info(self.reports_dir)
        self.best_model_name = str(self.best_meta.get("best_model", "xgboost")).lower()

        self.model_wrapper, self.checkpoint_path = load_best_model_checkpoint(self.best_model_name, self.experiments_dir)
        self.X_test, self.y_test, self.X_train, self.feature_names = load_processed_test_data()

        # Align feature columns if checkpoint metadata specifies feature_cols
        if isinstance(self.model_wrapper, dict) and "feature_cols" in self.model_wrapper:
            ckpt_features = self.model_wrapper["feature_cols"]
            common_feats = [c for c in ckpt_features if c in self.X_test.columns]
            if len(common_feats) == len(ckpt_features):
                self.feature_names = ckpt_features
                self.X_test = self.X_test[self.feature_names]
                self.X_train = self.X_train[self.feature_names]

        self.explainer = None
        self.shap_values = None
        self.shap_matrix: Optional[np.ndarray] = None
        self.expected_value: float = 0.0

    def _extract_underlying_estimator(self) -> Any:
        """Extracts the underlying fitted scikit-learn / XGBoost / LightGBM estimator object."""
        m = self.model_wrapper

        if isinstance(m, dict):
            if "model" in m and m["model"] is not None:
                m = m["model"]
            elif "models" in m and isinstance(m["models"], dict):
                models_dict = m["models"]
                if "P50" in models_dict:
                    m = models_dict["P50"]
                elif 0.5 in models_dict:
                    m = models_dict[0.5]
                else:
                    m = list(models_dict.values())[0]
            elif "estimator" in m and m["estimator"] is not None:
                m = m["estimator"]

        if hasattr(m, "model") and m.model is not None:
            return m.model
        elif hasattr(m, "models") and isinstance(m.models, dict):
            if "P50" in m.models:
                return m.models["P50"]
            elif 0.5 in m.models:
                return m.models[0.5]
            else:
                return list(m.models.values())[0]

        return m

    def _predict_array(self, X_input: Any) -> np.ndarray:
        """Helper to ensure input is formatted as DataFrame and return 1D numpy predictions."""
        if isinstance(X_input, np.ndarray):
            df_in = pd.DataFrame(X_input, columns=self.feature_names)
        elif isinstance(X_input, pd.DataFrame):
            df_in = X_input
        else:
            df_in = pd.DataFrame(X_input)

        if hasattr(self.model_wrapper, "predict"):
            preds = self.model_wrapper.predict(df_in)
        else:
            estimator = self._extract_underlying_estimator()
            preds = estimator.predict(df_in)

        if isinstance(preds, dict):
            preds = preds.get("P50", list(preds.values())[0])

        return np.asarray(preds, dtype=np.float64).flatten()

    def initialize_explainer(self):
        """Initializes TreeExplainer or KernelExplainer based on model architecture."""
        estimator = self._extract_underlying_estimator()
        predict_fn = lambda x: self._predict_array(x)

        tree_models = ["xgboost", "random_forest", "lightgbm_quantile", "lightgbm"]
        is_tree = any(tm in self.best_model_name for tm in tree_models)

        if is_tree:
            logger.info(f"Initializing TreeExplainer for tree-based model: '{self.best_model_name}'")
            try:
                # Direct TreeExplainer call on underlying scikit-learn / XGBoost estimator
                self.explainer = shap.TreeExplainer(estimator)
            except Exception as e1:
                try:
                    booster = getattr(estimator, "get_booster", lambda: None)()
                    if booster is not None:
                        self.explainer = shap.TreeExplainer(booster)
                    else:
                        raise e1
                except Exception as e2:
                    logger.warning(f"TreeExplainer failed ({e2}), falling back to KernelExplainer...")
                    bg_sample = shap.sample(self.X_train, min(20, len(self.X_train)))
                    self.explainer = shap.KernelExplainer(predict_fn, bg_sample)
        else:
            logger.info(f"Initializing KernelExplainer for model: '{self.best_model_name}'")
            bg_sample = shap.sample(self.X_train, min(20, len(self.X_train)))
            self.explainer = shap.KernelExplainer(predict_fn, bg_sample)

    def compute_shap_values(self) -> shap.Explanation:
        """Computes SHAP values on X_test dataset."""
        if self.explainer is None:
            self.initialize_explainer()

        logger.info(f"Computing SHAP values on X_test (samples={len(self.X_test)})...")

        if isinstance(self.explainer, shap.KernelExplainer):
            eval_test = self.X_test.iloc[:min(100, len(self.X_test))]
            raw_shap = self.explainer.shap_values(eval_test, nsamples=50)
            target_test = eval_test
        else:
            target_test = self.X_test
            try:
                raw_shap = self.explainer(target_test)
            except Exception:
                raw_shap = self.explainer.shap_values(target_test)

        self.target_test = target_test

        # Convert to standard Explanation object if numpy array returned
        if isinstance(raw_shap, np.ndarray):
            self.shap_matrix = raw_shap
            exp_val = getattr(self.explainer, "expected_value", 0.0)
            if isinstance(exp_val, np.ndarray):
                exp_val = float(exp_val.flatten()[0])
            self.expected_value = float(exp_val)
            
            self.shap_values = shap.Explanation(
                values=self.shap_matrix,
                base_values=np.full(len(target_test), self.expected_value),
                data=target_test.values,
                feature_names=self.feature_names
            )
        elif isinstance(raw_shap, list):
            # Multi-output / quantile list handling
            self.shap_matrix = raw_shap[0]
            exp_val = getattr(self.explainer, "expected_value", 0.0)
            if isinstance(exp_val, (list, np.ndarray)):
                exp_val = float(exp_val[0])
            self.expected_value = float(exp_val)

            self.shap_values = shap.Explanation(
                values=self.shap_matrix,
                base_values=np.full(len(target_test), self.expected_value),
                data=target_test.values,
                feature_names=self.feature_names
            )
        else:
            self.shap_values = raw_shap
            self.shap_matrix = getattr(raw_shap, "values", np.array([]))
            exp_v = getattr(raw_shap, "base_values", 0.0)
            if isinstance(exp_v, (np.ndarray, list)):
                self.expected_value = float(np.mean(exp_v))
            else:
                self.expected_value = float(exp_v)

        logger.info(f"SHAP value computation complete. Matrix shape: {self.shap_matrix.shape}")
        return self.shap_values
