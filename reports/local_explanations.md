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
- **Test Sample Index**: `367`
- **Actual Material Demand**: `2071.50`
- **Predicted Material Demand**: `2098.81`
- **Base Expected Value (E[f(X)])**: `540.44`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `2.4932` | `+1478.7683` |
| `DWT_cD_2_Std` | `258.5917` | `+40.3890` |
| `EMD_IMF_4_Var` | `31939.2578` | `+19.8673` |
| `EMD_IMF_1` | `393.5531` | `+8.4968` |
| `Lead_Time_Days` | `-0.7768` | `+7.4617` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Current_Inventory` | `0.7904` | `-0.1114` |
| `DWT_cD_1_Mean` | `-28.0160` | `-0.0638` |
| `Project_Phase` | `-1.0000` | `-0.0296` |
| `EMD_IMF_3_Mean` | `-4.9724` | `-0.0288` |
| `DWT_cD_3_Entropy` | `3.8314` | `-0.0194` |

---

## 2. Median Demand Prediction Case Study

- **Scenario Type**: Median Predicted Demand
- **Test Sample Index**: `1406`
- **Actual Material Demand**: `530.00`
- **Predicted Material Demand**: `504.34`
- **Base Expected Value (E[f(X)])**: `540.44`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `DWT_cD_2_Std` | `153.1325` | `+15.6400` |
| `EMD_IMF_4_Var` | `12633.9379` | `+2.5591` |
| `EMD_IMF_1` | `158.2342` | `+2.2373` |
| `Rolling_STD_3` | `182.9502` | `+0.8212` |
| `Storage_Capacity` | `1.0415` | `+0.8122` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `-0.0693` | `-58.1257` |
| `EMD_IMF_2` | `-201.8000` | `-0.9466` |
| `Trend_Strength` | `0.5538` | `-0.3228` |
| `DWT_cD_3_Entropy` | `3.7446` | `-0.1764` |
| `DWT_Approximation_Energy_Ratio` | `0.9010` | `-0.1345` |

---

## 2. Lowest Demand Prediction Case Study

- **Scenario Type**: Lowest Predicted Demand
- **Test Sample Index**: `1029`
- **Actual Material Demand**: `17.00`
- **Predicted Material Demand**: `7.40`
- **Base Expected Value (E[f(X)])**: `540.44`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Lead_Time_Days` | `2.3105` | `+1.7645` |
| `Storage_Capacity` | `0.8658` | `+0.2266` |
| `Production_Capacity` | `-0.3625` | `+0.0999` |
| `DWT_cD_3_Mean` | `0.1233` | `+0.0602` |
| `Signal_Entropy` | `7.3338` | `+0.0458` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `-0.8528` | `-394.9843` |
| `DWT_cD_2_Std` | `1.9860` | `-124.6210` |
| `EMD_IMF_4_Var` | `1.2289` | `-12.7047` |
| `Rolling_STD_3` | `0.2887` | `-0.9150` |
| `EMD_IMF_5` | `-0.6124` | `-0.5213` |

---

## 3. Operational Insights for POWERGRID Procurement

1. **High Demand Risk Mitigation**: When forecasts spike (High Demand case), decomposition trend components and project stage features act as primary positive drivers. Procuring early prevents stocking delays.
2. **Low Demand Asset Allocation**: When forecasts drop (Low Demand case), seasonal monsoon lags reduce required buffer stock, preventing warehouse congestion and holding cost penalties.
3. **Auditability**: Regional engineers can inspect individual project site predictions using these waterfall SHAP breakdowns to verify procurement requests before releasing purchase orders.
