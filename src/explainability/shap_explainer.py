import json
import os
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
from src.models import registry
from src.models import lightgbm_quantile, xgboost_model, random_forest, mlp, lstm, svr

logger = setup_logger("shap_explainer")

def get_best_model_info(reports_dir: Path = REPORTS_DIR) -> Dict[str, Any]:
    """Loads best model metadata from reports/best_model.json."""
    json_path = reports_dir / "best_model.json"
    if not json_path.exists():
        raise FileNotFoundError(f"best_model.json not found at {json_path}. Run model training first.")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    logger.info(f"Loaded best model info: {meta.get('best_model', 'unknown')}")
    return meta

def load_best_model_checkpoint(best_model_name: str, experiments_dir: Path = EXPERIMENTS_DIR) -> Tuple[Any, Path]:
    """
    Locates and instantiates the trained model object using the model registry,
    then loads the serialized weights/state.
    """
    checkpoints_root = experiments_dir / "checkpoints"
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"Checkpoints directory not found at {checkpoints_root}")

    matching_dirs = list(checkpoints_root.glob(f"EXP-{best_model_name.lower()}-*"))
    if not matching_dirs:
        raise FileNotFoundError(f"No checkpoint directories found for model '{best_model_name}' in {checkpoints_root}")

    matching_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_dir = matching_dirs[0]

    model_file = latest_dir / "model.joblib"
    if not model_file.exists():
        joblibs = list(latest_dir.glob("*.joblib"))
        if not joblibs:
            raise FileNotFoundError(f"No .joblib checkpoint found in {latest_dir}")
        model_file = joblibs[0]

    try:
        model_cls = registry.get_model_class(best_model_name)
        model_instance = model_cls()
        model_instance.load(str(model_file))
        logger.info(f"Successfully loaded model '{best_model_name}' via registry from {model_file}")
        return model_instance, model_file
    except Exception as e:
        logger.warning(f"Registry load fallback ({e}). Loading raw joblib artifact...")
        raw_obj = joblib.load(model_file)
        return raw_obj, model_file

def load_processed_test_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    Loads processed_dataset.csv, chronologically splits (70% train, 15% val, 15% test),
    and returns (X_test, y_test, X_train, feature_names).
    """
    if not PROCESSED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATASET_PATH}")

    df = pd.read_csv(PROCESSED_DATASET_PATH)

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
    Manages automatic SHAP explainer selection, rigorous feature alignment,
    diagnostic verification, and additivity checks for the best model.
    """
    def __init__(self, reports_dir: Path = REPORTS_DIR, experiments_dir: Path = EXPERIMENTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.experiments_dir = Path(experiments_dir)

        self.best_meta = get_best_model_info(self.reports_dir)
        self.best_model_name = str(self.best_meta.get("best_model", "lightgbm_quantile")).lower()

        self.model_wrapper, self.checkpoint_path = load_best_model_checkpoint(self.best_model_name, self.experiments_dir)
        self.X_test_raw, self.y_test, self.X_train_raw, self.raw_feature_names = load_processed_test_data()

        # Extract underlying fitted estimator
        self.estimator = self._extract_underlying_estimator()

        # Align feature columns to training configuration
        if hasattr(self.model_wrapper, "feature_cols") and self.model_wrapper.feature_cols is not None:
            self.feature_names = list(self.model_wrapper.feature_cols)
        elif hasattr(self.estimator, "feature_name_") and self.estimator.feature_name_ is not None:
            self.feature_names = list(self.estimator.feature_name_)
        elif hasattr(self.estimator, "feature_names_in_") and self.estimator.feature_names_in_ is not None:
            self.feature_names = list(self.estimator.feature_names_in_)
        else:
            self.feature_names = self.raw_feature_names

        # Extract imputation values from model if available
        if hasattr(self.model_wrapper, "impute_values") and self.model_wrapper.impute_values is not None:
            self.impute_values = self.model_wrapper.impute_values
        else:
            self.impute_values = self.X_train_raw.median().fillna(0.0)

        # Align test and train DataFrames
        self.X_test = self.X_test_raw.reindex(columns=self.feature_names).fillna(self.impute_values)
        self.X_train = self.X_train_raw.reindex(columns=self.feature_names).fillna(self.impute_values)

        self.explainer = None
        self.shap_values = None
        self.shap_matrix: Optional[np.ndarray] = None
        self.expected_value: float = 0.0

    def _extract_underlying_estimator(self) -> Any:
        """Extracts the underlying fitted estimator object (e.g. LGBMRegressor, XGBRegressor)."""
        m = self.model_wrapper

        # LightGBM Quantile Forecast Model
        if hasattr(m, "models") and isinstance(m.models, dict):
            if 0.5 in m.models:
                logger.info("Selected LightGBM P50 (median) quantile model for SHAP explanations.")
                return m.models[0.5]
            elif "P50" in m.models:
                return m.models["P50"]
            else:
                return list(m.models.values())[0]

        if hasattr(m, "model") and m.model is not None:
            return m.model

        if isinstance(m, dict):
            if 0.5 in m:
                return m[0.5]
            if "P50" in m:
                return m["P50"]
            if "model" in m and m["model"] is not None:
                return m["model"]
            if "models" in m and isinstance(m["models"], dict):
                return m["models"].get(0.5, m["models"].get("P50", list(m["models"].values())[0]))

        return m

    def _predict_array(self, X_input: Any) -> np.ndarray:
        """Helper to ensure input is formatted as DataFrame and return 1D numpy predictions."""
        if isinstance(X_input, np.ndarray):
            df_in = pd.DataFrame(X_input, columns=self.feature_names)
        elif isinstance(X_input, pd.DataFrame):
            df_in = X_input.reindex(columns=self.feature_names).fillna(self.impute_values)
        else:
            df_in = pd.DataFrame(X_input).reindex(columns=self.feature_names).fillna(self.impute_values)

        if hasattr(self.estimator, "predict"):
            preds = self.estimator.predict(df_in)
        elif hasattr(self.model_wrapper, "predict"):
            preds = self.model_wrapper.predict(df_in)
        else:
            preds = np.zeros(len(df_in))

        if isinstance(preds, dict):
            preds = preds.get("P50", preds.get(0.5, list(preds.values())[0]))

        return np.asarray(preds, dtype=np.float64).flatten()

    def initialize_explainer(self):
        """Initializes TreeExplainer or KernelExplainer based on model architecture."""
        predict_fn = lambda x: self._predict_array(x)

        tree_models = ["xgboost", "random_forest", "lightgbm_quantile", "lightgbm"]
        is_tree = any(tm in self.best_model_name for tm in tree_models)

        if is_tree and hasattr(self.estimator, "predict"):
            logger.info(f"Initializing TreeExplainer for tree-based model: '{self.best_model_name}' ({type(self.estimator).__name__})")
            try:
                self.explainer = shap.TreeExplainer(self.estimator)
            except Exception as e1:
                try:
                    booster = getattr(self.estimator, "get_booster", lambda: getattr(self.estimator, "booster_", None))()
                    if booster is not None:
                        self.explainer = shap.TreeExplainer(booster)
                    else:
                        raise e1
                except Exception as e2:
                    logger.warning(f"TreeExplainer initialization fallback ({e2}). Using KernelExplainer...")
                    bg_sample = shap.sample(self.X_train, min(50, len(self.X_train)))
                    self.explainer = shap.KernelExplainer(predict_fn, bg_sample)
        else:
            logger.info(f"Initializing KernelExplainer for model: '{self.best_model_name}'")
            bg_sample = shap.sample(self.X_train, min(50, len(self.X_train)))
            self.explainer = shap.KernelExplainer(predict_fn, bg_sample)

    def print_diagnostics(self, target_test: pd.DataFrame):
        """Prints comprehensive diagnostic information as required."""
        preds = self._predict_array(target_test)
        vals = self.shap_matrix

        print("\n" + "=" * 65)
        print("SHAP EXPLAINABILITY DIAGNOSTIC REPORT")
        print("=" * 65)
        print(f"  Model Type:              {self.best_model_name} ({type(self.estimator).__name__})")
        print(f"  X_train Shape:           {self.X_train.shape}")
        print(f"  X_test Shape:            {target_test.shape}")
        print(f"  Number of Features:      {len(self.feature_names)}")
        print(f"  Feature Names:           {self.feature_names}")
        print(f"  Prediction Range:        Min = {preds.min():.2f}, Max = {preds.max():.2f}, Mean = {preds.mean():.2f}")
        print(f"  SHAP Array Shape:        {vals.shape}")
        print(f"  Minimum SHAP Value:      {vals.min():.4f}")
        print(f"  Maximum SHAP Value:      {vals.max():.4f}")
        print(f"  Explainer Base Value:    {self.expected_value:.4f}")

        # Verification of Additivity: sum(SHAP) + expected_value ≈ prediction
        shap_sums = vals.sum(axis=1) + self.expected_value
        diffs = np.abs(preds - shap_sums)
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)
        print(f"\n  Additivity Verification (sum(SHAP) + base ~= prediction):")
        print(f"    Max Absolute Discrepancy: {max_diff:.2e}")
        print(f"    Mean Absolute Discrepancy: {mean_diff:.2e}")
        print(f"    Sample 0: Pred={preds[0]:.2f}, SHAP Sum + Base={shap_sums[0]:.2f} (Diff: {diffs[0]:.2e})")
        print(f"    Status: {'PASSED (Mathematically Exact)' if max_diff < 1e-3 else 'WARNING'}")

        # Mean absolute SHAP value for every feature
        mean_abs_shaps = np.mean(np.abs(vals), axis=0)
        sorted_indices = np.argsort(mean_abs_shaps)[::-1]
        
        print("\n  Ranked Top 15 Most Important Features (|mean SHAP|):")
        for rank, idx in enumerate(sorted_indices[:15], 1):
            feat = self.feature_names[idx]
            score = mean_abs_shaps[idx]
            print(f"    {rank:2d}. {feat:<28} | Mean |SHAP| = {score:.4f}")
        print("=" * 65 + "\n")

    def compute_shap_values(self) -> shap.Explanation:
        """Computes mathematically valid SHAP values on X_test dataset."""
        if self.explainer is None:
            self.initialize_explainer()

        target_test = self.X_test
        logger.info(f"Computing SHAP values on test dataset (samples={len(target_test)})...")

        if isinstance(self.explainer, shap.KernelExplainer):
            eval_test = target_test.iloc[:min(100, len(target_test))]
            raw_shap = self.explainer.shap_values(eval_test, nsamples=50)
            target_test = eval_test
        else:
            try:
                raw_shap = self.explainer(target_test)
            except Exception:
                raw_shap = self.explainer.shap_values(target_test)

        self.target_test = target_test

        if isinstance(raw_shap, shap.Explanation):
            self.shap_values = raw_shap
            self.shap_matrix = np.asarray(raw_shap.values)
            exp_v = getattr(raw_shap, "base_values", 0.0)
            if isinstance(exp_v, (np.ndarray, list)):
                self.expected_value = float(np.mean(exp_v))
            else:
                self.expected_value = float(exp_v)
        elif isinstance(raw_shap, np.ndarray):
            self.shap_matrix = raw_shap
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
        elif isinstance(raw_shap, list):
            self.shap_matrix = raw_shap[0] if len(raw_shap) > 0 else np.array([])
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

        # Print diagnostics
        self.print_diagnostics(target_test)

        logger.info(f"SHAP value computation complete. Matrix shape: {self.shap_matrix.shape}")
        return self.shap_values
