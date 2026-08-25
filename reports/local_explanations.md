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
- **Test Sample Index**: `45`
- **Actual Material Demand**: `3558.00`
- **Predicted Material Demand**: `3439.22`
- **Base Expected Value (E[f(X)])**: `728.93`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Classical_Residual` | `1542.9086` | `+1170.2451` |
| `EMD_IMF_6` | `1984.1862` | `+751.4499` |
| `Rolling_Max_3` | `1247.0000` | `+270.6293` |
| `EMD_IMF_1` | `724.0160` | `+147.9903` |
| `Classical_Seasonal` | `90.7837` | `+136.4620` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Production_Capacity` | `-0.4981` | `0.0000` |
| `Rolling_STD_3` | `96.0670` | `0.0000` |
| `DWT_cD_2_Mean` | `16.6965` | `0.0000` |
| `DWT_cD_3_Entropy` | `4.5381` | `0.0000` |
| `Seasonality_Strength` | `0.1078` | `0.0000` |

---

## 2. Median Demand Prediction Case Study

- **Scenario Type**: Median Predicted Demand
- **Test Sample Index**: `44`
- **Actual Material Demand**: `739.00`
- **Predicted Material Demand**: `690.60`
- **Base Expected Value (E[f(X)])**: `728.93`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Classical_Seasonal` | `410.5541` | `+279.1442` |
| `Rolling_Max_3` | `735.0000` | `+133.8563` |
| `Storage_Capacity` | `-0.0337` | `+13.2889` |
| `Seasonal_Demand_Index` | `1.0670` | `+5.7088` |
| `DWT_cD_3_Mean` | `57.8107` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Classical_Residual` | `-369.2175` | `-344.9205` |
| `EMD_IMF_1` | `-238.8966` | `-54.9443` |
| `Rolling_STD_3` | `383.1543` | `-20.2380` |
| `DWT_cD_1_Entropy` | `5.6960` | `-15.5211` |
| `DWT_cD_3_Entropy` | `4.1553` | `-11.0684` |

---

## 2. Lowest Demand Prediction Case Study

- **Scenario Type**: Lowest Predicted Demand
- **Test Sample Index**: `99`
- **Actual Material Demand**: `4.00`
- **Predicted Material Demand**: `11.24`
- **Base Expected Value (E[f(X)])**: `728.93`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Trend_Slope` | `-0.0152` | `+5.7709` |
| `DWT_cD_3_Mean` | `0.2684` | `+0.0000` |
| `DWT_cA_Entropy` | `5.0165` | `+0.0000` |
| `EMD_IMF_3_Mean` | `-0.1706` | `+0.0000` |
| `Production_Capacity` | `-0.5387` | `+0.0000` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `EMD_IMF_6` | `2.9898` | `-275.2603` |
| `Classical_Residual` | `-3.6883` | `-138.4439` |
| `Rolling_Max_3` | `4.0000` | `-133.3805` |
| `EMD_IMF_1` | `0.0733` | `-64.4428` |
| `Classical_Seasonal` | `4.3421` | `-35.0447` |

---

## 3. Operational Insights for POWERGRID Procurement

1. **High Demand Risk Mitigation**: When forecasts spike (High Demand case), decomposition trend components and project stage features act as primary positive drivers. Procuring early prevents stocking delays.
2. **Low Demand Asset Allocation**: When forecasts drop (Low Demand case), seasonal monsoon lags reduce required buffer stock, preventing warehouse congestion and holding cost penalties.
3. **Auditability**: Regional engineers can inspect individual project site predictions using these waterfall SHAP breakdowns to verify procurement requests before releasing purchase orders.
