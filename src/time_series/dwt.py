import pywt
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from src.utils.helpers import setup_logger

logger = setup_logger("dwt_transformer")

class DWTTransformer:
    """
    Discrete Wavelet Transform (DWT) decomposition helper:
    - Supports configurable wavelet families (default 'db4')
    - Performs multi-level decomposition
    - Extracts approximation (cA) and detail (cD) coefficients
    - Fits and serializes decomposition results per group
    """
    def __init__(self, wavelet: str = "db4", level: int = 3, save_path: Optional[Union[str, Path]] = None):
        self.wavelet = wavelet
        self.level = level
        self.save_path = Path(save_path) if save_path else None
        self.decomposition_results: Dict[Tuple[Any, ...], Dict[str, np.ndarray]] = {}
        self.is_fit = False

    def decompose_signal(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Performs multi-level wavelet decomposition on a single 1D signal.
        """
        signal_arr = np.array(signal, dtype=np.float64)
        
        # Calculate max level supported for this signal length and wavelet
        try:
            w = pywt.Wavelet(self.wavelet)
            max_level = pywt.dwt_max_level(len(signal_arr), w)
        except Exception as e:
            logger.warning(f"Error checking max wavelet level: {e}. Falling back to default.")
            max_level = self.level
            
        effective_level = min(self.level, max_level)
        if effective_level <= 0:
            effective_level = 1  # Fallback to level 1 if signal is very short
            
        logger.debug(f"Decomposing signal of length {len(signal_arr)} at level {effective_level} using {self.wavelet}")
        
        # wavedec returns [cA_n, cD_n, cD_n-1, ..., cD_1]
        coeffs = pywt.wavedec(signal_arr, wavelet=self.wavelet, level=effective_level)
        
        results = {
            "cA": coeffs[0]
        }
        for idx in range(1, len(coeffs)):
            results[f"cD_{len(coeffs) - idx}"] = coeffs[idx]
            
        return results

    def fit_transform(
        self,
        df: pd.DataFrame,
        group_cols: Optional[List[str]] = None,
        date_col: str = "Date",
        target_col: str = "Quantity_Required"
    ) -> Dict[Tuple[Any, ...], Dict[str, np.ndarray]]:
        """
        Fits DWT on the dataset: performs decomposition per group and stores the coefficients.
        """
        logger.info(f"Fitting DWTTransformer (wavelet: {self.wavelet}, level: {self.level})...")
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
        logger.info(f"DWT decomposition completed for {len(self.decomposition_results)} groups.")
        
        if self.save_path:
            self.save()
            
        return self.decomposition_results

    def save(self, filepath: Optional[Union[str, Path]] = None):
        """Serializes the DWTTransformer state to disk."""
        path = Path(filepath) if filepath else self.save_path
        if not path:
            raise ValueError("No save path specified.")
            
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "wavelet": self.wavelet,
            "level": self.level,
            "decomposition_results": self.decomposition_results,
            "is_fit": self.is_fit
        }
        joblib.dump(state, path)
        logger.info(f"DWTTransformer saved to {path}")

    def load(self, filepath: Union[str, Path]):
        """Loads serialized DWTTransformer state from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No DWT file found at {path}")
            
        state = joblib.load(path)
        self.wavelet = state["wavelet"]
        self.level = state["level"]
        self.decomposition_results = state["decomposition_results"]
        self.is_fit = state["is_fit"]
        logger.info(f"DWTTransformer loaded from {path}")
