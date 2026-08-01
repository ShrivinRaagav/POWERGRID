# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented production forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | WMAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| xgboost | 150.320743 | 218.515741 | 30.6741% | 27.68% | 42.8239% | 0.709838 | 2.0531 | 0.0143 | N/A |
| lightgbm_quantile | 174.127621 | 243.542956 | 32.5352% | 32.07% | 44.2587% | 0.639565 | 4.6809 | 0.0237 | 50.063676 |
| random_forest | 157.834466 | 253.719658 | 29.7015% | 29.07% | 38.9991% | 0.608814 | 1.9854 | 0.0366 | N/A |
| svr | 212.771794 | 296.245494 | 41.3705% | 39.19% | 57.5088% | 0.466691 | 0.4343 | 0.1946 | N/A |
| lstm | 430.951906 | 550.128394 | 177.2375% | 79.37% | 116.0091% | -0.839093 | 31.0485 | 0.6893 | N/A |
| mlp | 352.260413 | 588.592233 | 54.2689% | 64.88% | 72.0046% | -1.105255 | 1.9415 | 0.0112 | N/A |

---

## Best Performing Model

Model:
xgboost

Selection Criterion:
Lowest RMSE on the held-out chronological test dataset.

Performance Summary:
- RMSE: 218.515741
- MAE: 150.320743
- WMAPE: 27.68%
- R²: 0.709838

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
