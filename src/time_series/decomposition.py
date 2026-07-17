import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from statsmodels.tsa.seasonal import seasonal_decompose

from src.config.settings import (
    TS_DWT_WAVELET, TS_DWT_LEVEL, TS_DWT_OUTPUT_PATH,
    TS_EMD_SPLINE, TS_EMD_MAX_IMFS, TS_EMD_STD_THR, TS_EMD_ENERGY_RATIO_THR, TS_EMD_OUTPUT_PATH,
    TS_FEATURES_OUTPUT_PATH, TS_SUMMARY_PATH, TS_GROUP_BY, TS_DATE_COL, TS_TARGET_COL
)
from src.time_series.utils import prepare_time_series
from src.time_series.dwt import DWTTransformer
from src.time_series.emd import EMDProcessor
from src.utils.helpers import setup_logger

logger = setup_logger("decomposition_extractor")

class TimeSeriesFeatureExtractor:
    """
    Orchestrates time-series decomposition (Classical, DWT, and EMD)
    and extracts statistical/signal features.
    """
    def __init__(
        self,
        group_cols: Optional[List[str]] = TS_GROUP_BY,
        date_col: str = TS_DATE_COL,
        target_col: str = TS_TARGET_COL,
        save_path: Optional[Union[str, Path]] = TS_FEATURES_OUTPUT_PATH
    ):
        self.group_cols = group_cols
        self.date_col = date_col
        self.target_col = target_col
        self.save_path = Path(save_path) if save_path else None
        
        # Instantiate sub-processors
        self.dwt_transformer = DWTTransformer(
            wavelet=TS_DWT_WAVELET,
            level=TS_DWT_LEVEL,
            save_path=TS_DWT_OUTPUT_PATH
        )
        self.emd_processor = EMDProcessor(
            spline_kind=TS_EMD_SPLINE,
            max_imf=TS_EMD_MAX_IMFS,
            save_path=TS_EMD_OUTPUT_PATH,
            std_thr=TS_EMD_STD_THR,
            energy_ratio_thr=TS_EMD_ENERGY_RATIO_THR
        )
        
        self.extracted_features: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.is_fit = False

    def _get_entropy(self, x: np.ndarray) -> float:
        """Calculates Shannon entropy of normalized signal energy."""
        x_sq = x ** 2
        total_energy = np.sum(x_sq)
        if total_energy < 1e-12:
            return 0.0
        p = x_sq / total_energy
        return -np.sum(p * np.log2(p + 1e-12))

    def _get_dominant_frequency(self, x: np.ndarray) -> float:
        """Finds dominant frequency using FFT."""
        # Subtract mean to remove DC component
        x_detrended = x - np.mean(x)
        n = len(x_detrended)
        if n < 2:
            return 0.0
        fft_vals = np.fft.fft(x_detrended)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.fftfreq(n)
        
        # Only consider positive frequencies
        pos_mask = freqs > 0
        if not np.any(pos_mask):
            return 0.0
        psd_pos = psd[pos_mask]
        freqs_pos = freqs[pos_mask]
        
        return freqs_pos[np.argmax(psd_pos)]

    def _get_trend_slope(self, x: np.ndarray) -> float:
        """Fits a linear trend line and returns the slope."""
        n = len(x)
        if n < 2:
            return 0.0
        t = np.arange(n)
        slope, _ = np.polyfit(t, x, 1)
        return float(slope)

    def extract_group_features(
        self,
        signal: np.ndarray,
        dwt_coeffs: Dict[str, np.ndarray],
        emd_results: Dict[str, np.ndarray],
        classical_trend: np.ndarray,
        classical_seasonal: np.ndarray,
        classical_resid: np.ndarray
    ) -> Dict[str, Any]:
        """
        Extracts features from DWT, EMD, and Classical decomposition for a single signal.
        """
        feats = {}
        n = len(signal)
        feats["Signal_Length"] = n
        
        # Basic Signal Energy & Entropy
        signal_energy = np.sum(signal ** 2)
        feats["Signal_Energy"] = signal_energy
        feats["Signal_Mean_Energy"] = np.mean(signal ** 2)
        feats["Signal_Entropy"] = self._get_entropy(signal)
        feats["Dominant_Frequency"] = self._get_dominant_frequency(signal)
        
        # Classical Decomposition Strengths
        var_resid = np.var(classical_resid) if len(classical_resid) > 0 else 0.0
        var_trend_resid = np.var(classical_trend + classical_resid) if len(classical_trend) > 0 else 0.0
        var_seas_resid = np.var(classical_seasonal + classical_resid) if len(classical_seasonal) > 0 else 0.0
        
        feats["Trend_Strength"] = max(0.0, 1.0 - var_resid / (var_trend_resid + 1e-12))
        feats["Seasonality_Strength"] = max(0.0, 1.0 - var_resid / (var_seas_resid + 1e-12))
        
        # Classical Trend & Residual stats
        feats["Trend_Slope"] = self._get_trend_slope(classical_trend)
        feats["Residual_Variance"] = var_resid
        feats["Residual_Mean"] = np.mean(classical_resid)
        feats["Residual_Std"] = np.std(classical_resid)
        
        # DWT features
        # Approximation coefficients (cA)
        cA = dwt_coeffs["cA"]
        cA_energy = np.sum(cA ** 2)
        feats["DWT_cA_Energy"] = cA_energy
        feats["DWT_cA_Mean"] = np.mean(cA)
        feats["DWT_cA_Std"] = np.std(cA)
        feats["DWT_cA_Var"] = np.var(cA)
        feats["DWT_cA_Entropy"] = self._get_entropy(cA)
        
        # Detail coefficients (cD_i)
        cD_keys = [k for k in dwt_coeffs.keys() if k.startswith("cD_")]
        total_dwt_energy = cA_energy
        detail_energy = 0.0
        
        for k in cD_keys:
            cD_vals = dwt_coeffs[k]
            cD_energy = np.sum(cD_vals ** 2)
            detail_energy += cD_energy
            total_dwt_energy += cD_energy
            
            feats[f"DWT_{k}_Energy"] = cD_energy
            feats[f"DWT_{k}_Mean"] = np.mean(cD_vals)
            feats[f"DWT_{k}_Std"] = np.std(cD_vals)
            feats[f"DWT_{k}_Var"] = np.var(cD_vals)
            feats[f"DWT_{k}_Entropy"] = self._get_entropy(cD_vals)
            
        feats["DWT_Approximation_Energy_Ratio"] = cA_energy / (total_dwt_energy + 1e-12)
        feats["DWT_Detail_Energy_Ratio"] = detail_energy / (total_dwt_energy + 1e-12)
        
        # EMD features
        imfs = emd_results["imfs"]
        residual = emd_results["residual"]
        num_imfs = imfs.shape[0]
        feats["Num_IMFs"] = num_imfs
        
        imf_energies = []
        for i in range(num_imfs):
            imf = imfs[i]
            imf_energy = np.sum(imf ** 2)
            imf_energies.append(imf_energy)
            
            feats[f"EMD_IMF_{i+1}_Energy"] = imf_energy
            feats[f"EMD_IMF_{i+1}_Mean"] = np.mean(imf)
            feats[f"EMD_IMF_{i+1}_Std"] = np.std(imf)
            feats[f"EMD_IMF_{i+1}_Var"] = np.var(imf)
            
        # Dominant IMF
        feats["Dominant_IMF"] = int(np.argmax(imf_energies) + 1) if imf_energies else 0
        
        # EMD Residual stats
        feats["EMD_Resid_Energy"] = np.sum(residual ** 2)
        feats["EMD_Resid_Mean"] = np.mean(residual)
        feats["EMD_Resid_Std"] = np.std(residual)
        feats["EMD_Resid_Var"] = np.var(residual)
        feats["EMD_Resid_Slope"] = self._get_trend_slope(residual)
        
        return feats

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes complete decomposition and feature extraction.
        Returns:
        - df_enriched: Copy of the processed dataset with EMD, DWT, and classical signals and features added.
        - df_summary: A summary dataset containing one row per time series group.
        """
        logger.info("Starting Time-Series Decomposition Pipeline...")
        
        # 1. Format dataset as chronological time series
        df_ts = prepare_time_series(df, group_cols=self.group_cols, date_col=self.date_col, target_col=self.target_col)
        
        # 2. Run DWT and EMD decompositions
        dwt_results = self.dwt_transformer.fit_transform(df_ts, group_cols=self.group_cols, date_col=self.date_col, target_col=self.target_col)
        emd_results = self.emd_processor.fit_transform(df_ts, group_cols=self.group_cols, date_col=self.date_col, target_col=self.target_col)
        
        # 3. Process each group
        summary_rows = []
        df_enriched_list = []
        
        df_sorted = df_ts.sort_values(by=[self.date_col]).copy()
        
        if not self.group_cols:
            grouped = [((), df_sorted)]
        else:
            grouped = df_sorted.groupby(self.group_cols)
            
        for name, group in grouped:
            key = name if isinstance(name, tuple) else (name,)
            signal = group[self.target_col].values
            
            # Classical decomposition (additive)
            n_obs = len(signal)
            # Default period 52 (yearly for weekly data). Fallback to smaller periods if short.
            period = min(52, n_obs // 2)
            if period >= 2:
                decomp = seasonal_decompose(signal, period=period, model='additive', extrapolate_trend='freq')
                c_trend = decomp.trend
                c_seasonal = decomp.seasonal
                c_resid = decomp.resid
            else:
                c_trend = signal.copy()
                c_seasonal = np.zeros_like(signal)
                c_resid = np.zeros_like(signal)
                
            # Grab DWT and EMD results
            d_res = dwt_results[key]
            e_res = emd_results[key]
            
            # Extract features
            group_feats = self.extract_group_features(
                signal=signal,
                dwt_coeffs=d_res,
                emd_results=e_res,
                classical_trend=c_trend,
                classical_seasonal=c_seasonal,
                classical_resid=c_resid
            )
            
            self.extracted_features[key] = group_feats
            
            # Add metadata keys to summary row
            summary_row = {}
            if self.group_cols:
                if len(self.group_cols) == 1:
                    summary_row[self.group_cols[0]] = name
                else:
                    for i, col in enumerate(self.group_cols):
                        summary_row[col] = name[i]
            summary_row.update(group_feats)
            summary_rows.append(summary_row)
            
            # Enrich the group time series with time-varying signals
            group_enriched = group.copy()
            
            # Add Classical components
            group_enriched["Classical_Trend"] = c_trend
            group_enriched["Classical_Seasonal"] = c_seasonal
            group_enriched["Classical_Residual"] = c_resid
            
            # Add EMD IMFs and residual
            e_imfs = e_res["imfs"]
            for i in range(e_imfs.shape[0]):
                group_enriched[f"EMD_IMF_{i+1}"] = e_imfs[i]
            group_enriched["EMD_Residual"] = e_res["residual"]
            
            # Add static group features (broadcasting them to all dates in this group)
            for k, v in group_feats.items():
                # Avoid duplicates
                if k not in group_enriched.columns:
                    group_enriched[k] = v
                    
            df_enriched_list.append(group_enriched)
            
        # Combine enriched dataframes
        df_enriched = pd.concat(df_enriched_list, ignore_index=True)
        # Sort chronologically
        if self.group_cols:
            df_enriched = df_enriched.sort_values(by=[self.date_col] + self.group_cols).reset_index(drop=True)
        else:
            df_enriched = df_enriched.sort_values(by=[self.date_col]).reset_index(drop=True)
            
        # Build summary report
        df_summary = pd.DataFrame(summary_rows)
        # Ensure directories exist and save
        TS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(TS_SUMMARY_PATH, index=False)
        logger.info(f"Summary report generated and saved to {TS_SUMMARY_PATH}")
        
        self.is_fit = True
        if self.save_path:
            self.save()
            
        return df_enriched, df_summary

    def save(self, filepath: Optional[Union[str, Path]] = None):
        """Serializes the TimeSeriesFeatureExtractor state to disk."""
        path = Path(filepath) if filepath else self.save_path
        if not path:
            raise ValueError("No save path specified.")
            
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "group_cols": self.group_cols,
            "date_col": self.date_col,
            "target_col": self.target_col,
            "extracted_features": self.extracted_features,
            "is_fit": self.is_fit
        }
        joblib.dump(state, path)
        logger.info(f"TimeSeriesFeatureExtractor saved to {path}")

    def load(self, filepath: Union[str, Path]):
        """Loads serialized TimeSeriesFeatureExtractor state from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No Feature Extractor file found at {path}")
            
        state = joblib.load(path)
        self.group_cols = state["group_cols"]
        self.date_col = state["date_col"]
        self.target_col = state["target_col"]
        self.extracted_features = state["extracted_features"]
        self.is_fit = state["is_fit"]
        logger.info(f"TimeSeriesFeatureExtractor loaded from {path}")
