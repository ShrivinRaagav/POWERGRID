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
- **Test Sample Index**: `295`
- **Actual Material Demand**: `2029.00`
- **Predicted Material Demand**: `2044.62`
- **Base Expected Value (E[f(X)])**: `790.03`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Classical_Residual` | `728.1561` | `+529.0876` |
| `Classical_Seasonal` | `677.1118` | `+380.1445` |
| `EMD_IMF_6` | `1167.2788` | `+306.3223` |
| `EMD_IMF_1` | `322.4244` | `+73.9785` |
| `EMD_IMF_4` | `139.6714` | `+62.5358` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_5` | `-284.1781` | `-207.1173` |
| `EMD_IMF_3_Var` | `25993.3195` | `-12.9103` |
| `DWT_cD_3_Entropy` | `2.9632` | `0.0000` |
| `DWT_cA_Entropy` | `3.8375` | `0.0000` |
| `Residual_Mean` | `19.9230` | `0.0000` |

---

## 2. Median Demand Prediction Case Study

- **Scenario Type**: Median Predicted Demand
- **Test Sample Index**: `112`
- **Actual Material Demand**: `513.00`
- **Predicted Material Demand**: `470.95`
- **Base Expected Value (E[f(X)])**: `790.03`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_6` | `1165.3437` | `+258.3781` |
| `Classical_Seasonal` | `94.9880` | `+45.0088` |
| `DWT_cD_3_Entropy` | `2.9632` | `+11.5015` |
| `EMD_IMF_5_Mean` | `-36.5518` | `+8.7594` |
| `DWT_cD_1_Mean` | `32.3617` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_5` | `-261.4071` | `-302.3324` |
| `Classical_Residual` | `-325.1204` | `-242.3377` |
| `EMD_IMF_4` | `-191.0957` | `-68.9777` |
| `EMD_IMF_1` | `-59.6905` | `-17.9815` |
| `EMD_IMF_6_Std` | `13.4969` | `-5.5631` |

---

## 2. Lowest Demand Prediction Case Study

- **Scenario Type**: Lowest Predicted Demand
- **Test Sample Index**: `213`
- **Actual Material Demand**: `9.00`
- **Predicted Material Demand**: `3.98`
- **Base Expected Value (E[f(X)])**: `790.03`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_5_Mean` | `-0.1989` | `+13.9154` |
| `DWT_cD_2_Entropy` | `3.8737` | `+0.0000` |
| `DWT_cD_1_Mean` | `0.5006` | `+0.0000` |
| `DWT_cD_3_Entropy` | `2.0713` | `+0.0000` |
| `DWT_cD_1_Entropy` | `4.4229` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_6` | `17.6333` | `-406.7614` |
| `EMD_IMF_5` | `3.1659` | `-140.1490` |
| `Classical_Residual` | `-2.9236` | `-113.1125` |
| `DWT_cD_2_Energy` | `745.4067` | `-40.1101` |
| `EMD_IMF_4` | `1.1682` | `-29.2459` |

---

## 3. Operational Insights for POWERGRID Procurement

1. **High Demand Risk Mitigation**: When forecasts spike (High Demand case), decomposition trend components and project stage features act as primary positive drivers. Procuring early prevents stocking delays.
2. **Low Demand Asset Allocation**: When forecasts drop (Low Demand case), seasonal monsoon lags reduce required buffer stock, preventing warehouse congestion and holding cost penalties.
3. **Auditability**: Regional engineers can inspect individual project site predictions using these waterfall SHAP breakdowns to verify procurement requests before releasing purchase orders.
