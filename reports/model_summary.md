# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented production forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | WMAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| lightgbm_quantile | 17.945315 | 51.574281 | 11.9436% | 3.03% | 13.0000% | 0.987161 | 2.0031 | 0.0155 | 15.970153 |
| xgboost | 36.673402 | 66.922507 | 16.1151% | 6.18% | 20.2390% | 0.978382 | 2.0343 | 0.0109 | N/A |
| random_forest | 33.236503 | 74.195468 | 14.0006% | 5.60% | 16.9940% | 0.973428 | 3.4948 | 0.0236 | N/A |
| mlp | 61.088810 | 89.483026 | 34.6107% | 10.30% | 23.5588% | 0.961350 | 2.8367 | 0.0080 | N/A |
| lstm | 169.598459 | 233.839203 | 80.9159% | 28.59% | 43.6129% | 0.736062 | 46.4352 | 0.2168 | N/A |
| svr | 227.747367 | 314.195321 | 48.3686% | 38.39% | 57.3578% | 0.523497 | 1.7583 | 0.8317 | N/A |

---

## Best Performing Model

Model:
lightgbm_quantile

Selection Criterion:
Lowest RMSE on the held-out chronological test dataset.

Performance Summary:
- RMSE: 51.574281
- MAE: 17.945315
- WMAPE: 3.03%
- R²: 0.987161

Future Usage:
This model will be forwarded to:
- Module 3.5 (Forecast Model Evaluation)
- Module 4 (SHAP Explainability)
- Module 5 (Multi-Objective Supply Chain Optimization)

---

## Metric Documentation

- **MAE**: Average absolute prediction error.
- **RMSE**: Penalizes larger prediction errors more heavily.
- **MAPE**: Average percentage prediction error.
- **WMAPE**: Weighted Mean Absolute Percentage Error. Recommended for demand forecasting.
- **SMAPE**: Symmetric percentage error between actual and predicted values.
- **R²**: Coefficient of determination measuring explained variance.
- **Pinball Loss**: Measures quality of probabilistic quantile predictions (LightGBM Quantile only).
