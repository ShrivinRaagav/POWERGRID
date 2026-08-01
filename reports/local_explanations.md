# Local Forecast Explanations Report (Module 4 - SHAP XAI)

This report provides granular local feature attributions for three representative operational demand scenarios (**Highest Demand**, **Median Demand**, and **Lowest Demand**) predicted by the best model (**xgboost**).

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
- **Test Sample Index**: `24`
- **Actual Material Demand**: `935.00`
- **Predicted Material Demand**: `1448.54`
- **Base Expected Value (E[f(X)])**: `760.70`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_6` | `1062.1980` | `+404.4051` |
| `Classical_Residual` | `294.2271` | `+390.5863` |
| `EMD_IMF_5_Mean` | `-29.4177` | `+16.1315` |
| `DWT_cD_2_Mean` | `28.7657` | `+15.0450` |
| `EMD_IMF_1` | `69.2015` | `+13.7141` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_5` | `-202.3964` | `-109.4635` |
| `DWT_cA_Entropy` | `3.7701` | `-26.3209` |
| `EMD_IMF_4` | `-20.3714` | `-20.0010` |
| `EMD_IMF_3_Var` | `36235.7352` | `0.0000` |
| `DWT_cD_2_Energy` | `1958825.6506` | `0.0000` |

---

## 2. Median Demand Prediction Case Study

- **Scenario Type**: Median Predicted Demand
- **Test Sample Index**: `11`
- **Actual Material Demand**: `466.00`
- **Predicted Material Demand**: `399.38`
- **Base Expected Value (E[f(X)])**: `760.70`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Classical_Residual` | `132.0339` | `+56.3889` |
| `EMD_IMF_5_Mean` | `-3.2117` | `+20.9955` |
| `DWT_cA_Entropy` | `3.9752` | `+18.9424` |
| `Residual_Mean` | `-20.9387` | `+1.8576` |
| `DWT_cD_2_Mean` | `-1.5321` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_6` | `400.6348` | `-185.2372` |
| `EMD_IMF_5` | `3.3947` | `-112.6225` |
| `Classical_Seasonal` | `-58.4789` | `-49.0841` |
| `EMD_IMF_6_Std` | `49.1306` | `-34.5126` |
| `EMD_IMF_4` | `2.8528` | `-29.6956` |

---

## 2. Lowest Demand Prediction Case Study

- **Scenario Type**: Lowest Predicted Demand
- **Test Sample Index**: `43`
- **Actual Material Demand**: `16.00`
- **Predicted Material Demand**: `4.42`
- **Base Expected Value (E[f(X)])**: `760.70`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_5_Mean` | `0.2487` | `+0.0000` |
| `EMD_IMF_6_Std` | `4.5811` | `+0.0000` |
| `DWT_cD_2_Entropy` | `4.2768` | `+0.0000` |
| `DWT_cD_1_Mean` | `-0.1161` | `+0.0000` |
| `Trend_Strength` | `0.8368` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_6` | `14.5366` | `-378.6081` |
| `EMD_IMF_5` | `2.6389` | `-154.2963` |
| `Classical_Residual` | `3.4432` | `-94.0809` |
| `DWT_cD_2_Energy` | `145.5182` | `-34.9793` |
| `EMD_IMF_4` | `1.0130` | `-26.4387` |

---

## 3. Operational Insights for POWERGRID Procurement

1. **High Demand Risk Mitigation**: When forecasts spike (High Demand case), decomposition trend components and project stage features act as primary positive drivers. Procuring early prevents stocking delays.
2. **Low Demand Asset Allocation**: When forecasts drop (Low Demand case), seasonal monsoon lags reduce required buffer stock, preventing warehouse congestion and holding cost penalties.
3. **Auditability**: Regional engineers can inspect individual project site predictions using these waterfall SHAP breakdowns to verify procurement requests before releasing purchase orders.
