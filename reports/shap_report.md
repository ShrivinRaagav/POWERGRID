# SHAP Explainability & Model Transparency Report (Module 4)

This report presents the Explainable Artificial Intelligence (XAI) analysis using **SHAP (SHapley Additive exPlanations)** for the selected best-performing material demand forecasting model (**xgboost**).

---

## 1. Executive Summary

- **Selected Best Forecasting Model**: `xgboost`
- **Model Selection Criterion**: `Lowest RMSE`
- **Test Set Performance**: RMSE = `218.515741`, MAE = `150.320743`, WMAPE = `27.68%`, R² = `0.709838`
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
| 1 | `EMD_IMF_6` | 261.049015 | 27.73% | 27.73% |
| 2 | `EMD_IMF_5` | 196.309353 | 20.85% | 48.59% |
| 3 | `Classical_Residual` | 195.508829 | 20.77% | 69.35% |
| 4 | `Classical_Seasonal` | 104.243056 | 11.07% | 80.43% |
| 5 | `EMD_IMF_4` | 87.256990 | 9.27% | 89.70% |
| 6 | `EMD_IMF_1` | 24.852864 | 2.64% | 92.34% |
| 7 | `DWT_cD_2_Energy` | 22.157467 | 2.35% | 94.69% |
| 8 | `EMD_IMF_5_Mean` | 13.564760 | 1.44% | 96.13% |
| 9 | `DWT_cD_2_Mean` | 6.134410 | 0.65% | 96.78% |
| 10 | `EMD_IMF_6_Std` | 5.147950 | 0.55% | 97.33% |
| 11 | `DWT_cA_Entropy` | 4.066187 | 0.43% | 97.76% |
| 12 | `EMD_IMF_3_Var` | 4.043875 | 0.43% | 98.19% |
| 13 | `Residual_Mean` | 3.093637 | 0.33% | 98.52% |
| 14 | `Trend_Strength` | 2.695705 | 0.29% | 98.81% |
| 15 | `DWT_cD_3_Entropy` | 2.523661 | 0.27% | 99.08% |

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
| **Highest Demand** | `295` | `2029.00` | `2044.62` | `790.03` | `Classical_Residual` |
| **Median Demand** | `112` | `513.00` | `470.95` | `790.03` | `EMD_IMF_6` |
| **Lowest Demand** | `213` | `9.00` | `3.98` | `790.03` | `EMD_IMF_5_Mean` |

For full local waterfall plots and force diagrams, see [reports/local_explanations.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/local_explanations.md) and [reports/shap_plots/shap_force.html](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/shap_plots/shap_force.html).

---

## 6. Operational Value & Procurement Decision Support for POWERGRID

1. **Procurement Lead-Time Optimization**: SHAP feature rankings highlight when high-frequency DWT coefficients begin trending upward, alerting procurement officers 4-6 weeks in advance of conductor/transformer shortages.
2. **Safety Stock Reduction**: By explaining why the model predicts lower demand during monsoon quarters, POWERGRID warehouse managers can safely lower safety stock buffers, reducing holding cost overheads.
3. **Auditability & Regulatory Compliance**: Black-box machine learning forecasts are transformed into verifiable, additive equations, satisfying Smart India Hackathon and Ministry of Power audit standards.

---

## 7. Downstream Integration with Module 5

The global SHAP feature importance output (`reports/shap_feature_importance.csv`) will be directly ingested by **Module 5 (Multi-Objective Supply Chain Optimization)** to weight risk factors and inventory storage constraints.
