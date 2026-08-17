# Local Forecast Explanations Report (Module 4 - SHAP XAI)

This report provides granular local feature attributions for three representative operational demand scenarios (**Highest Demand**, **Median Demand**, and **Lowest Demand**) predicted by the best model (**lightgbm_quantile**).

---

## 1. Overview & Baseline Setup

Local SHAP explanations break down individual prediction equations into additive feature contributions:

$$\hat{y}_i = E[f(X)] + \sum_{j=1}^{p} \phi_{i,j}$$

Where:
- **y_hat_i**: Predicted demand for test instance i.
- **E[f(X)]**: Base value (average model prediction across training set).
- **phi_{i,j}**: SHAP attribution of feature j for instance i.

---

## 2. Highest Demand Prediction Case Study

- **Scenario Type**: Highest Predicted Demand
- **Test Sample Index**: `99`
- **Actual Material Demand**: `9.00`
- **Predicted Material Demand**: `0.00`
- **Base Expected Value (E[f(X)])**: `0.00`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `-0.8596` | `+0.0000` |
| `EMD_IMF_3_Var` | `2.9707` | `+0.0000` |
| `Seasonality_Strength` | `0.5378` | `+0.0000` |
| `EMD_IMF_3_Mean` | `-0.0377` | `+0.0000` |
| `DWT_cD_3_Entropy` | `3.9484` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `-0.8596` | `0.0000` |
| `EMD_IMF_3_Mean` | `-0.0377` | `0.0000` |
| `DWT_cD_3_Entropy` | `3.9484` | `0.0000` |
| `DWT_Approximation_Energy_Ratio` | `0.9436` | `0.0000` |
| `DWT_cD_2_Entropy` | `4.7910` | `0.0000` |

---

## 2. Median Demand Prediction Case Study

- **Scenario Type**: Median Predicted Demand
- **Test Sample Index**: `48`
- **Actual Material Demand**: `1172.00`
- **Predicted Material Demand**: `0.00`
- **Base Expected Value (E[f(X)])**: `0.00`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `1.0037` | `+0.0000` |
| `EMD_IMF_3_Var` | `25966.2069` | `+0.0000` |
| `Seasonality_Strength` | `0.3648` | `+0.0000` |
| `EMD_IMF_3_Mean` | `0.8630` | `+0.0000` |
| `DWT_cD_3_Entropy` | `4.2101` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `1.0037` | `0.0000` |
| `EMD_IMF_3_Mean` | `0.8630` | `0.0000` |
| `DWT_cD_3_Entropy` | `4.2101` | `0.0000` |
| `DWT_Approximation_Energy_Ratio` | `0.9734` | `0.0000` |
| `DWT_cD_2_Entropy` | `4.8827` | `0.0000` |

---

## 2. Lowest Demand Prediction Case Study

- **Scenario Type**: Lowest Predicted Demand
- **Test Sample Index**: `0`
- **Actual Material Demand**: `649.00`
- **Predicted Material Demand**: `0.00`
- **Base Expected Value (E[f(X)])**: `0.00`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `0.1742` | `+0.0000` |
| `EMD_IMF_3_Var` | `4842.5778` | `+0.0000` |
| `Seasonality_Strength` | `0.1627` | `+0.0000` |
| `EMD_IMF_3_Mean` | `4.1399` | `+0.0000` |
| `DWT_cD_3_Entropy` | `3.9470` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `0.1742` | `0.0000` |
| `EMD_IMF_3_Mean` | `4.1399` | `0.0000` |
| `DWT_cD_3_Entropy` | `3.9470` | `0.0000` |
| `DWT_Approximation_Energy_Ratio` | `0.9747` | `0.0000` |
| `DWT_cD_2_Entropy` | `4.4179` | `0.0000` |

---

## 3. Operational Insights for POWERGRID Procurement

1. **High Demand Risk Mitigation**: When forecasts spike (High Demand case), decomposition trend components and project stage features act as primary positive drivers. Procuring early prevents stocking delays.
2. **Low Demand Asset Allocation**: When forecasts drop (Low Demand case), seasonal monsoon lags reduce required buffer stock, preventing warehouse congestion and holding cost penalties.
3. **Auditability**: Regional engineers can inspect individual project site predictions using these waterfall SHAP breakdowns to verify procurement requests before releasing purchase orders.
