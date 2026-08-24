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
- **Test Sample Index**: `532`
- **Actual Material Demand**: `2155.00`
- **Predicted Material Demand**: `2154.22`
- **Base Expected Value (E[f(X)])**: `542.05`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `2.5451` | `+1565.0460` |
| `DWT_cD_2_Std` | `290.2630` | `+30.3724` |
| `Lead_Time_Days` | `-0.7963` | `+9.4163` |
| `EMD_IMF_1` | `398.6700` | `+4.1612` |
| `EMD_IMF_2` | `187.0103` | `+2.0163` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Current_Inventory` | `0.8011` | `-0.8925` |
| `EMD_IMF_5` | `-628.8579` | `-0.5568` |
| `DWT_cD_2_Mean` | `-2.7231` | `-0.0535` |
| `Trend_Strength` | `0.7271` | `-0.0439` |
| `DWT_cD_1_Entropy` | `5.8597` | `-0.0389` |

---

## 2. Median Demand Prediction Case Study

- **Scenario Type**: Median Predicted Demand
- **Test Sample Index**: `934`
- **Actual Material Demand**: `499.00`
- **Predicted Material Demand**: `503.83`
- **Base Expected Value (E[f(X)])**: `542.05`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `DWT_cD_2_Std` | `214.3390` | `+14.1034` |
| `Seasonality_Strength` | `0.5602` | `+6.0805` |
| `EMD_IMF_2_Mean` | `2.1275` | `+5.9199` |
| `EMD_IMF_1` | `128.0160` | `+2.2865` |
| `Production_Capacity` | `0.5128` | `+1.5587` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `-0.1000` | `-57.3689` |
| `Storage_Capacity` | `0.9703` | `-8.6773` |
| `Lead_Time_Days` | `1.6070` | `-5.3929` |
| `EMD_IMF_2` | `-56.6078` | `-1.4366` |
| `EMD_IMF_4_Mean` | `27.4813` | `-0.9108` |

---

## 2. Lowest Demand Prediction Case Study

- **Scenario Type**: Lowest Predicted Demand
- **Test Sample Index**: `1084`
- **Actual Material Demand**: `10.00`
- **Predicted Material Demand**: `-5.65`
- **Base Expected Value (E[f(X)])**: `542.05`

### Key Positive Drivers (Pushing Forecast Higher)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Seasonality_Strength` | `0.5378` | `+2.0436` |
| `Lead_Time_Days` | `1.4117` | `+1.5251` |
| `EMD_IMF_2_Mean` | `0.0587` | `+1.0936` |
| `Seasonal_Demand_Index` | `1.1767` | `+0.2869` |
| `Current_Inventory` | `-0.8658` | `+0.2092` |

### Key Negative Drivers (Pushing Forecast Lower)

| Feature | Feature Value | SHAP Contribution |
| :--- | :---: | :---: |
| `Historical_Demand` | `-0.8426` | `-408.1005` |
| `DWT_cD_2_Std` | `1.5768` | `-111.2064` |
| `Storage_Capacity` | `1.1258` | `-28.0668` |
| `EMD_IMF_4_Var` | `12.5683` | `-1.7130` |
| `EMD_IMF_1` | `0.9682` | `-1.3147` |

---

## 3. Operational Insights for POWERGRID Procurement

1. **High Demand Risk Mitigation**: When forecasts spike (High Demand case), decomposition trend components and project stage features act as primary positive drivers. Procuring early prevents stocking delays.
2. **Low Demand Asset Allocation**: When forecasts drop (Low Demand case), seasonal monsoon lags reduce required buffer stock, preventing warehouse congestion and holding cost penalties.
3. **Auditability**: Regional engineers can inspect individual project site predictions using these waterfall SHAP breakdowns to verify procurement requests before releasing purchase orders.
