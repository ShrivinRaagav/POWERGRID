# POWERGRID Decomposition Quality Report

This report summarizes the mathematical and signal processing quality of the Discrete Wavelet Transform (DWT) and Empirical Mode Decomposition (EMD) modules fitted on the processed POWERGRID dataset.

---

## 📊 Summary Statistics

| Metric | DWT (Wavelet) | EMD (Sifting) |
| :--- | :---: | :---: |
| **Wavelet Used / Spline Type** | `db4` | `cubic` |
| **Decomposition Levels / Max IMFs** | `3` | `5` |
| **Average Reconstruction RMSE** | `0.000000` | `0.000000` |
| **Average Reconstruction Correlation** | `1.000000` | `1.000000` |
| **Number of Time Series Groups** | `21` | `21` |

---

## 🧠 Decomposition Insights

### 1. Empirical Mode Decomposition (EMD)
- **Intrinsic Mode Functions (IMFs)**: EMD extracts dynamic, data-driven oscillatory modes.
  - **Min IMFs extracted**: 5
  - **Max IMFs extracted**: 6
  - **Average IMFs extracted**: 5.76
- **Signal Reconstructibility**: The sifting process successfully decomposes demand. EMD reconstruction ($S_t = \sum IMF_i + R_t$) achieves an average correlation of **100.0000%** with the original signal.

### 2. Discrete Wavelet Transform (DWT)
- **Wavelet Multi-Resolution Analysis (MRA)**: Decomposes the signal into an approximation level (cA) and detail levels (cD_1 to cD_3).
- **Signal Reconstructibility**: Inverse DWT (`pywt.waverec`) achieves an average correlation of **100.0000%** with the original signal, confirming zero-loss numeric transformation.

### 3. Supply Chain Dynamics
- **Average Trend Strength ($F_T$)**: `0.6070`
  - Represents the proportion of demand variance explained by the low-frequency project baseline.
- **Average Seasonality Strength ($F_S$)**: `0.3774`
  - Represents the strength of cyclic variations (monsoons, holiday slowdowns, weather waves) within the dataset.

---

## 🔬 Interpretation of Results

1. **Perfect Wavelet Reconstruction**: DWT reconstruction yields negligible RMSE (essentially float precision limits). This indicates that the extracted approximation and detail coefficients capture all frequency information without any distortion.
2. **High-Fidelity EMD**: EMD reconstruction error is exceptionally low (with correlation very close to 1.0). Minor discrepancies represent sub-threshold numerical residuals from the cubic spline envelope fitting process, which does not impact forecasting utility.
3. **Optimized Forecasting Inputs**: The trend, seasonality, IMFs, and wavelet coefficients provide orthogonal, stabilized representations of demand. Downstream ML models (e.g. LSTM, XGBoost) can leverage these components to capture different timescale behaviors (long-term expansion trends vs short-term weather delays) separately, reducing forecast error.
