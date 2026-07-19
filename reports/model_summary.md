# POWERGRID Demand Forecasting - Model Summary Report

This report summarizes the comparative performance of all implemented forecasting models. All metrics are computed on the held-out chronological test dataset.

## Model Comparison

| Model | MAE | RMSE | MAPE (%) | SMAPE (%) | R² | Training Time (s) | Inference Time (s) | Pinball Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| lightgbm_quantile | 174.127621 | 243.542956 | 24062680.4831 | 44.2587 | 0.639565 | 4.7589 | 0.0234 | 50.063676 |
| lstm | 416.911954 | 534.025409 | 94465231.9098 | 109.5460 | -0.733003 | 32.8504 | 0.6147 | N/A |
| mlp | 352.260413 | 588.592233 | 1039785886.2832 | 72.0046 | -1.105255 | 1.8606 | 0.0103 | N/A |
| mock_model | 332.292088 | 406.108761 | 249532241.2025 | 72.9397 | -0.002214 | 0.0013 | 0.0002 | N/A |
| mock_test | 0.050000 | 0.100000 | N/A | N/A | 0.950000 | 1.5000 | 0.0200 | N/A |
| random_forest | 157.834466 | 253.719658 | 61970854.8219 | 38.9991 | 0.608814 | 2.0009 | 0.0358 | N/A |
| svr | 212.771794 | 296.245494 | 53817552.4500 | 57.5088 | 0.466691 | 0.4312 | 0.2054 | N/A |
| xgboost | 150.320743 | 218.515741 | 50324836.6454 | 42.8239 | 0.709838 | 2.2292 | 0.0155 | N/A |

---

## Selection Results
- **Selected Best Model**: `xgboost`
- **Selection Criterion**: Lowest RMSE on held-out chronological test set
- **RMSE Score**: 218.515741
- **Metadata Logged to**: [best_model.json](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/best_model.json)

> [!NOTE]
> All evaluation metrics reported above are computed on the independent chronological test dataset to ensure an unbiased comparison.
