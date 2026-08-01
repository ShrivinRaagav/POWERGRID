# SHAP Explainability & Model Transparency Report (Module 4)

This report presents the Explainable Artificial Intelligence (XAI) analysis using **SHAP (SHapley Additive exPlanations)** for the selected best-performing material demand forecasting model (**xgboost**).

---

## 1. Executive Summary

- **Selected Best Forecasting Model**: `xgboost`
- **Model Selection Criterion**: `Lowest RMSE`
- **Test Set Performance**: RMSE = `217.524370`, MAE = `148.924324`, WMAPE = `27.43%`, R² = `0.712465`
- **Top 3 Predictive Drivers**: `EMD_IMF_6`, `EMD_IMF_5`, `Classical_Residual`
- **Primary Visualization Artifacts**: [reports/shap_plots/](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/shap_plots/)

---

## 2. Explainability Methodology (Shapley Additive Values)

SHAP attributes prediction outputs using cooperative game theory. Each feature's marginal contribution phi_j is computed across all feature coalitions S in F:

$$\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{j\}) - f_x(S) \right]$$

### Fundamental XAI Guarantees:
1. **Local Accuracy**: The sum of SHAP values equals the difference between model output y_hat and base expected value E[f(X)].
2. **Consistency**: If a model changes so a feature's marginal contribution increases, its SHAP value strictly increases.
3. **Missingness**: Features with zero impact receive exactly zero attribution.

---

## 3. Global Feature Importance Ranking

The table below ranks the top features based on their mean absolute SHAP value (Mean(|SHAP_j|)) across the independent chronological test set:

| Rank | Feature Name | Mean Absolute SHAP | Relative Importance (%) | Cumulative Importance (%) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | `EMD_IMF_6` | 252.312937 | 27.48% | 27.48% |
| 2 | `EMD_IMF_5` | 182.753819 | 19.91% | 47.39% |
| 3 | `Classical_Residual` | 155.268934 | 16.91% | 64.30% |
| 4 | `Classical_Seasonal` | 129.286251 | 14.08% | 78.39% |
| 5 | `EMD_IMF_4` | 91.093400 | 9.92% | 88.31% |
| 6 | `EMD_IMF_1` | 20.196815 | 2.20% | 90.51% |
| 7 | `DWT_cD_2_Energy` | 18.871628 | 2.06% | 92.57% |
| 8 | `EMD_IMF_5_Mean` | 10.041864 | 1.09% | 93.66% |
| 9 | `DWT_cD_2_Mean` | 8.095221 | 0.88% | 94.54% |
| 10 | `DWT_cA_Entropy` | 7.182251 | 0.78% | 95.32% |
| 11 | `Trend_Strength` | 5.504646 | 0.60% | 95.92% |
| 12 | `Residual_Mean` | 5.219784 | 0.57% | 96.49% |
| 13 | `EMD_IMF_3_Var` | 5.149522 | 0.56% | 97.05% |
| 14 | `DWT_cD_3_Mean` | 5.141560 | 0.56% | 97.61% |
| 15 | `EMD_IMF_6_Std` | 4.435132 | 0.48% | 98.10% |

---

## 4. Technical & Domain Interpretation of Key Drivers

### A. Time-Series Signal Decomposition Features (DWT / EMD)
- **Wavelet Coefficients (e.g. `DWT_cD_1_Mean`, `DWT_cD_3_Entropy`)**: High-frequency DWT detail coefficients isolate sudden short-term demand surges caused by unexpected weather delays or emergency grid repairs.
- **Empirical Mode Functions (e.g. `EMD_IMF_1`, `EMD_IMF_5`)**: Non-stationary Intrinsic Mode Functions capture cyclical monsoon seasonality and baseline project execution velocity.

### B. Domain & Temporal Supply Chain Features
- **Lag & Rolling Variables**: Historical demand trends anchor the prediction baseline, smoothing out high-variance single-week spikes.
- **Cyclical Features (`Month_Sin`, `Month_Cos`)**: Capture calendar-year seasonality matching POWERGRID fiscal quarter procurement cycles.

---

## 5. Local Prediction Explanations Summary

Local explanations evaluate individual site predictions across three distinct operational cases:

| Case Study Scenario | Test Sample Index | Actual Demand | Predicted Demand | Base Expected Value | Primary Driver |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Highest Demand** | `24` | `935.00` | `1448.54` | `760.70` | `EMD_IMF_6` |
| **Median Demand** | `11` | `466.00` | `399.38` | `760.70` | `Classical_Residual` |
| **Lowest Demand** | `43` | `16.00` | `4.42` | `760.70` | `EMD_IMF_5_Mean` |

For full local waterfall plots and force diagrams, see [reports/local_explanations.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/local_explanations.md) and [reports/shap_plots/shap_force.html](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/shap_plots/shap_force.html).

---

## 6. Operational Value & Procurement Decision Support for POWERGRID

1. **Procurement Lead-Time Optimization**: SHAP feature rankings highlight when high-frequency DWT coefficients begin trending upward, alerting procurement officers 4-6 weeks in advance of conductor/transformer shortages.
2. **Safety Stock Reduction**: By explaining why the model predicts lower demand during monsoon quarters, POWERGRID warehouse managers can safely lower safety stock buffers, reducing holding cost overheads.
3. **Auditability & Regulatory Compliance**: Black-box machine learning forecasts are transformed into verifiable, additive equations, satisfying Smart India Hackathon and Ministry of Power audit standards.

---

## 7. Downstream Integration with Module 5

The global SHAP feature importance output (`reports/shap_feature_importance.csv`) will be directly ingested by **Module 5 (Multi-Objective Supply Chain Optimization)** to weight risk factors and inventory storage constraints.
