# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented production forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | WMAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| lightgbm_quantile | 19.326147 | 52.295281 | 14.9395% | 3.26% | 18.1941% | 0.986799 | 3.1759 | 0.0402 | 9.661683 |
| xgboost | 36.673402 | 66.922507 | 16.1151% | 6.18% | 20.2390% | 0.978382 | 2.5203 | 0.0259 | N/A |
| random_forest | 33.236503 | 74.195468 | 14.0006% | 5.60% | 16.9940% | 0.973428 | 8.4493 | 0.0536 | N/A |
| mlp | 61.088810 | 89.483026 | 34.6107% | 10.30% | 23.5588% | 0.961350 | 7.5479 | 0.0184 | N/A |
| lstm | 208.481620 | 280.415240 | 113.6253% | 35.15% | 50.5477% | 0.620449 | 74.2729 | 0.5500 | N/A |
| svr | 227.747367 | 314.195321 | 48.3686% | 38.39% | 57.3578% | 0.523497 | 4.4755 | 2.1722 | N/A |

---

## Best Performing Model

Model:
lightgbm_quantile

Selection Criterion:
Lowest RMSE on the held-out chronological test dataset.

Performance Summary:
- RMSE: 52.295281
- MAE: 19.326147
- WMAPE: 3.26%
- R²: 0.986799

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
