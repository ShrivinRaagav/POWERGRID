import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

from src.config.settings import (
    FS_VARIANCE_THR, FS_CORRELATION_THR, FS_IMPORTANCE_METHOD, FS_TOP_K,
    FS_KEEP_COLS, FS_SELECTOR_PATH, FS_CORRELATION_PATH, FS_REPORT_PATH
)
from src.feature_selection.variance import find_low_variance_features
from src.feature_selection.correlation import find_highly_correlated_features
from src.feature_selection.importance import calculate_feature_importances
from src.feature_selection.utils import find_duplicated_features
from src.utils.helpers import setup_logger

logger = setup_logger("feature_selector")

class FeatureSelector:
    """
    Orchestrates the entire feature selection process:
    1. Removes duplicated features.
    2. Removes constant and low-variance features.
    3. Calculates feature importances (Mutual Info or Random Forest).
    4. Removes highly correlated features, retaining the one with higher importance.
    5. Retains the top K features, while ensuring protected/mandatory columns are kept.
    6. Generates feature selection reports.
    """
    def __init__(
        self,
        variance_threshold: float = FS_VARIANCE_THR,
        correlation_threshold: float = FS_CORRELATION_THR,
        importance_method: str = FS_IMPORTANCE_METHOD,
        top_k: int = FS_TOP_K,
        keep_cols: List[str] = FS_KEEP_COLS,
        save_path: Optional[Union[str, Path]] = FS_SELECTOR_PATH,
        corr_save_path: Optional[Union[str, Path]] = FS_CORRELATION_PATH,
        report_path: Optional[Union[str, Path]] = FS_REPORT_PATH
    ):
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.importance_method = importance_method
        self.top_k = top_k
        self.keep_cols = keep_cols
        
        self.save_path = Path(save_path) if save_path else None
        self.corr_save_path = Path(corr_save_path) if corr_save_path else None
        self.report_path = Path(report_path) if report_path else None
        
        self.selected_features: List[str] = []
        self.dropped_features: Dict[str, str] = {} # Map feature name -> drop reason
        self.feature_metadata: Dict[str, Dict[str, Any]] = {} # Map feature -> metadata metrics
        self.is_fit = False

    def fit(self, df: pd.DataFrame, target_col: str = "Quantity_Required"):
        """
        Fits the feature selection filters on the training dataset.
        """
        logger.info("Fitting FeatureSelector on training dataset...")
        self.dropped_features = {}
        self.feature_metadata = {}
        self.selected_features = []
        
        # 0. Initial feature set
        all_cols = list(df.columns)
        numerical_cols = list(df.select_dtypes(include=[np.number]).columns)
        candidate_features = [c for c in numerical_cols if c not in self.keep_cols and c != target_col]
        
        # Initialize metadata
        for col in all_cols:
            self.feature_metadata[col] = {
                "Variance": np.nan,
                "Importance": np.nan,
                "Correlation_Flag": False,
                "Correlation_Reason": "",
                "Drop_Reason": ""
            }
            
        # Protect keep columns from selection logic but ensure they are included in final select
        protected_selected = [c for c in self.keep_cols if c in all_cols]
        for col in protected_selected:
            self.feature_metadata[col]["Drop_Reason"] = "Protected"
            
        # 1. Drop duplicated columns
        dup_cols = find_duplicated_features(df, exclude_cols=self.keep_cols + [target_col])
        for col in dup_cols:
            self.dropped_features[col] = "Duplicated Column"
            self.feature_metadata[col]["Drop_Reason"] = "Duplicated Column"
            
        candidate_features = [c for c in candidate_features if c not in dup_cols]
        
        # 2. Drop constant and low variance columns
        low_var_cols, variances = find_low_variance_features(
            df,
            threshold=self.variance_threshold,
            exclude_cols=self.keep_cols + [target_col]
        )
        for col, var in variances.items():
            self.feature_metadata[col]["Variance"] = float(var)
            
        for col in low_var_cols:
            if col not in self.dropped_features: # Avoid overwriting duplicated reason
                reason = "Constant feature (variance = 0)" if variances[col] == 0.0 else f"Low variance ({variances[col]:.5f})"
                self.dropped_features[col] = reason
                self.feature_metadata[col]["Drop_Reason"] = reason
                
        candidate_features = [c for c in candidate_features if c not in low_var_cols]
        
        # 3. Calculate feature importances on remaining candidates
        # Create a temp dataframe with only targets and candidates to calculate importances
        importances = calculate_feature_importances(
            df=df,
            target_col=target_col,
            exclude_cols=self.keep_cols + list(self.dropped_features.keys()),
            method=self.importance_method
        )
        for col, imp in importances.items():
            self.feature_metadata[col]["Importance"] = float(imp)
            
        # 4. Filter highly correlated columns
        corr_cols, corr_matrix, pair_details = find_highly_correlated_features(
            df=df,
            threshold=self.correlation_threshold,
            exclude_cols=self.keep_cols + list(self.dropped_features.keys()) + [target_col],
            importances=importances,
            variances=variances
        )
        
        # Save correlation matrix if configured
        if self.corr_save_path and not corr_matrix.empty:
            self.corr_save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(corr_matrix, self.corr_save_path)
            logger.info(f"Correlation matrix saved to {self.corr_save_path}")
            
        for p in pair_details:
            drop_col = p["Drop"]
            self.feature_metadata[drop_col]["Correlation_Flag"] = True
            self.feature_metadata[drop_col]["Correlation_Reason"] = p["Reason"]
            
        for col in corr_cols:
            if col not in self.dropped_features:
                reason = self.feature_metadata[col]["Correlation_Reason"] or "High correlation"
                self.dropped_features[col] = reason
                self.feature_metadata[col]["Drop_Reason"] = reason
                
        candidate_features = [c for c in candidate_features if c not in corr_cols]
        
        # 5. Top-K Selection
        # Get importances for remaining candidates
        remaining_importances = {c: importances.get(c, 0.0) for c in candidate_features}
        sorted_candidates = sorted(remaining_importances.items(), key=lambda x: x[1], reverse=True)
        
        top_k_candidates = [c[0] for c in sorted_candidates[:self.top_k]]
        unselected_candidates = [c[0] for c in sorted_candidates[self.top_k:]]
        
        for col in unselected_candidates:
            rank = sorted_candidates.index((col, remaining_importances[col])) + 1
            reason = f"Ranked #{rank} in importance (not in top {self.top_k})"
            self.dropped_features[col] = reason
            self.feature_metadata[col]["Drop_Reason"] = reason
            
        # Final selected set: protected columns + top-K candidates
        self.selected_features = protected_selected + top_k_candidates
        
        for col in top_k_candidates:
            self.feature_metadata[col]["Drop_Reason"] = "Selected (Top Importance)"
            
        logger.info(f"Feature Selection complete: {len(self.selected_features)} features selected, {len(self.dropped_features)} features removed.")
        self.is_fit = True
        
        # 6. Generate and save report
        self.generate_report()
        
        if self.save_path:
            self.save()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the dataset: retains only selected features (and handles columns not in training).
        """
        if not self.is_fit:
            raise ValueError("FeatureSelector must be fitted before calling transform.")
            
        # Retain only columns that are selected AND present in the input df
        cols_to_keep = [c for c in self.selected_features if c in df.columns]
        
        # Also keep any other columns that are completely string/non-numeric columns if needed?
        # Typically the keep_cols handles dates and categoricals. So we strictly keep cols_to_keep.
        logger.info(f"Transforming dataset from shape {df.shape} to {df[cols_to_keep].shape}...")
        return df[cols_to_keep].copy()

    def generate_report(self):
        """Generates a detailed feature selection report CSV."""
        if not self.report_path:
            return
            
        rows = []
        for feat, meta in self.feature_metadata.items():
            importance = meta["Importance"]
            variance = meta["Variance"]
            drop_reason = meta["Drop_Reason"]
            
            selected = feat in self.selected_features
            
            rows.append({
                "Feature Name": feat,
                "Importance": f"{importance:.6f}" if not np.isnan(importance) else "N/A",
                "Correlation": meta["Correlation_Reason"] if meta["Correlation_Flag"] else "N/A",
                "Variance": f"{variance:.6f}" if not np.isnan(variance) else "N/A",
                "Status": "Selected" if selected else "Removed",
                "Reason": drop_reason if not selected else f"Kept ({drop_reason})"
            })
            
        df_report = pd.DataFrame(rows)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        df_report.to_csv(self.report_path, index=False)
        logger.info(f"Feature Selection report saved to {self.report_path}")

    def save(self, filepath: Optional[Union[str, Path]] = None):
        """Serializes the FeatureSelector state to disk."""
        path = Path(filepath) if filepath else self.save_path
        if not path:
            raise ValueError("No save path specified.")
            
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "variance_threshold": self.variance_threshold,
            "correlation_threshold": self.correlation_threshold,
            "importance_method": self.importance_method,
            "top_k": self.top_k,
            "keep_cols": self.keep_cols,
            "selected_features": self.selected_features,
            "dropped_features": self.dropped_features,
            "feature_metadata": self.feature_metadata,
            "is_fit": self.is_fit
        }
        joblib.dump(state, path)
        logger.info(f"FeatureSelector saved to {path}")

    def load(self, filepath: Union[str, Path]):
        """Loads serialized FeatureSelector state from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No Feature Selector file found at {path}")
            
        state = joblib.load(path)
        self.variance_threshold = state["variance_threshold"]
        self.correlation_threshold = state["correlation_threshold"]
        self.importance_method = state["importance_method"]
        self.top_k = state["top_k"]
        self.keep_cols = state["keep_cols"]
        self.selected_features = state["selected_features"]
        self.dropped_features = state["dropped_features"]
        self.feature_metadata = state["feature_metadata"]
        self.is_fit = state["is_fit"]
        logger.info(f"FeatureSelector loaded from {path}")
