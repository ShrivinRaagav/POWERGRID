from PyEMD import EMD
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from src.utils.helpers import setup_logger

logger = setup_logger("emd_processor")

class EMDProcessor:
    """
    Empirical Mode Decomposition (EMD) processor helper:
    - Uses PyEMD (installed as EMD-signal)
    - Generates Intrinsic Mode Functions (IMFs) and residual trend
    - Supports configurable stopping criteria via standard deviation, energy ratio, etc.
    - Fits and serializes decomposition results per group
    """
    def __init__(
        self,
        spline_kind: str = "cubic",
        max_imf: int = 5,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        self.spline_kind = spline_kind
        self.max_imf = max_imf
        self.save_path = Path(save_path) if save_path else None
        self.emd_params = kwargs
        self.decomposition_results: Dict[Tuple[Any, ...], Dict[str, np.ndarray]] = {}
        self.is_fit = False

    def decompose_signal(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Decomposes a single 1D signal into Intrinsic Mode Functions (IMFs) and residual trend.
        """
        signal_arr = np.array(signal, dtype=np.float64)
        
        # Check if the signal is virtually constant
        if np.std(signal_arr) < 1e-9:
            logger.warning("Signal has zero variance. Returning signal as a single IMF and zero residual.")
            imfs = np.expand_dims(signal_arr, axis=0)
            residual = np.zeros_like(signal_arr)
            return {
                "imfs": imfs,
                "residual": residual
            }
            
        # Instantiate EMD
        emd = EMD(spline_kind=self.spline_kind, **self.emd_params)
        
        try:
            imfs = emd(signal_arr, max_imf=self.max_imf)
            if imfs.ndim > 1:
                residual = signal_arr - np.sum(imfs, axis=0)
            else:
                residual = signal_arr - imfs
                imfs = np.expand_dims(imfs, axis=0)
        except Exception as e:
            logger.error(f"Error during EMD decomposition: {e}. Falling back to basic decomposition.")
            imfs = np.expand_dims(signal_arr, axis=0)
            residual = np.zeros_like(signal_arr)
            
        return {
            "imfs": imfs,
            "residual": residual
        }

    def fit_transform(
        self,
        df: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
        date_col: str = "Date",
        target_col: str = "Quantity_Required"
    ) -> Dict[Tuple[Any, ...], Dict[str, np.ndarray]]:
        """
        Fits EMD on the dataset: performs decomposition per group and stores IMFs & residuals.
        """
        logger.info(f"Fitting EMDProcessor (spline: {self.spline_kind}, max_imf: {self.max_imf})...")
        self.decomposition_results = {}
        
        df_sorted = df.sort_values(by=[date_col]).copy()
        
        if not group_cols:
            # Global time series
            key = ("global",)
            signal = df_sorted[target_col].values
            self.decomposition_results[key] = self.decompose_signal(signal)
        else:
            # Grouped time series
            grouped = df_sorted.groupby(group_cols)
            for name, group in grouped:
                key = name if isinstance(name, tuple) else (name,)
                signal = group[target_col].values
                self.decomposition_results[key] = self.decompose_signal(signal)
                
        self.is_fit = True
        logger.info(f"EMD decomposition completed for {len(self.decomposition_results)} groups.")
        
        if self.save_path:
            self.save()
            
        return self.decomposition_results

    def save(self, filepath: Optional[Union[str, Path]] = None):
        """Serializes the EMDProcessor state to disk."""
        path = Path(filepath) if filepath else self.save_path
        if not path:
            raise ValueError("No save path specified.")
            
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "spline_kind": self.spline_kind,
            "max_imf": self.max_imf,
            "emd_params": self.emd_params,
            "decomposition_results": self.decomposition_results,
            "is_fit": self.is_fit
        }
        joblib.dump(state, path)
        logger.info(f"EMDProcessor saved to {path}")

    def load(self, filepath: Union[str, Path]):
        """Loads serialized EMDProcessor state from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No EMD file found at {path}")
            
        state = joblib.load(path)
        self.spline_kind = state["spline_kind"]
        self.max_imf = state["max_imf"]
        self.emd_params = state["emd_params"]
        self.decomposition_results = state["decomposition_results"]
        self.is_fit = state["is_fit"]
        logger.info(f"EMDProcessor loaded from {path}")
