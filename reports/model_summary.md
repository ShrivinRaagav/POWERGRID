# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented production forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | WMAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| xgboost | 148.924324 | 217.524370 | 31.8701% | 27.43% | 42.5666% | 0.712465 | 0.3658 | 0.0171 | N/A |
| lightgbm_quantile | 174.127621 | 243.542956 | 32.5352% | 32.07% | 44.2587% | 0.639565 | 3.2490 | 0.0245 | 50.063676 |
| random_forest | 163.818430 | 265.158862 | 30.9400% | 30.17% | 39.9541% | 0.572744 | 1.9752 | 0.0354 | N/A |
| svr | 212.771794 | 296.245494 | 41.3705% | 39.19% | 57.5088% | 0.466691 | 0.4239 | 0.2064 | N/A |
| mlp | 416.371869 | 530.928348 | 83.3366% | 76.68% | 110.0581% | -0.712961 | 1.7501 | 0.0121 | N/A |
| lstm | 425.307788 | 540.141367 | 204.4658% | 78.33% | 112.8229% | -0.772925 | 29.3321 | 0.6516 | N/A |

---

## Best Performing Model

Model:
xgboost

Selection Criterion:
Lowest RMSE on the held-out chronological test dataset.

Performance Summary:
- RMSE: 217.524370
- MAE: 148.924324
- WMAPE: 27.43%
- R²: 0.712465

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
