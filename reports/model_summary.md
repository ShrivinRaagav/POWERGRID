# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented production forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | WMAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| xgboost | 65.992950 | 97.268501 | 36.4424% | 8.47% | 20.8115% | 0.983523 | 1.6817 | 0.0112 | N/A |
| mlp | 73.903709 | 110.635444 | 80.7869% | 9.48% | 29.3648% | 0.978683 | 1.4220 | 0.0082 | N/A |
| random_forest | 76.971371 | 116.865052 | 27.2532% | 9.88% | 18.9140% | 0.976215 | 3.3041 | 0.0246 | N/A |
| lightgbm_quantile | 75.206677 | 124.534498 | 18.6325% | 9.65% | 13.9855% | 0.972990 | 5.8851 | 0.1277 | 27.089460 |
| svr | 99.493466 | 241.176333 | 23.3916% | 12.76% | 16.9917% | 0.898700 | 1.7360 | 0.8101 | N/A |
| lstm | 337.150960 | 537.788679 | 223.8646% | 43.26% | 60.3956% | 0.496311 | 30.0520 | 0.2288 | N/A |

---

## Best Performing Model

Model:
xgboost

Selection Criterion:
Lowest RMSE on the held-out chronological test dataset.

Performance Summary:
- RMSE: 97.268501
- MAE: 65.992950
- WMAPE: 8.47%
- R²: 0.983523

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
