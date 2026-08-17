# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented production forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | WMAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| lightgbm_quantile | 17.837354 | 50.601806 | 12.4171% | 3.01% | 14.1606% | 0.987641 | 1.7137 | 0.0144 | 13.858858 |
| xgboost | 22.861951 | 53.818874 | 14.7513% | 3.85% | 20.1137% | 0.986019 | 1.5087 | 0.0081 | N/A |
| mlp | 27.111421 | 56.929308 | 20.8389% | 4.57% | 23.3806% | 0.984356 | 0.9983 | 0.0058 | N/A |
| random_forest | 23.356351 | 57.209336 | 13.8612% | 3.94% | 18.0402% | 0.984202 | 1.5059 | 0.0213 | N/A |
| lstm | 146.528785 | 228.924052 | 34.6956% | 24.70% | 30.3188% | 0.747041 | 23.7708 | 0.2130 | N/A |
| svr | 221.705513 | 313.886858 | 43.9116% | 37.38% | 54.9388% | 0.524432 | 1.4433 | 1.0168 | N/A |

---

## Best Performing Model

Model:
lightgbm_quantile

Selection Criterion:
Lowest RMSE on the held-out chronological test dataset.

Performance Summary:
- RMSE: 50.601806
- MAE: 17.837354
- WMAPE: 3.01%
- R²: 0.987641

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
