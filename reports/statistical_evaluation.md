# Statistical Evaluation & Model Significance Report (Module 3.5)

This report presents the rigorous statistical evaluation and pairwise significance analysis for the production forecasting models trained on the POWERGRID transmission material demand dataset.

---

## 1. Executive Summary

- **Primary Ranking Metric**: Root Mean Squared Error (RMSE)
- **Secondary Ranking Metrics**: Mean Absolute Error (MAE) & Weighted Mean Absolute Percentage Error (WMAPE)
- **Significance Level ($lpha$)**: 0.05
- **Overall Multi-Model Comparison**: Friedman Non-Parametric Test
- **Pairwise Model Comparison**: Wilcoxon Signed-Rank Test

---

## 2. Overall Multi-Model Comparison (Friedman Test)

The **Friedman Test** evaluates whether there is a statistically significant difference in performance across all evaluated forecasting models.

| Metric | Value |
| :--- | :--- |
| **Friedman $\chi^2$ Statistic** | `2994.8869` |
| **p-value** | `0.000000e+00` |
| **Degrees of Freedom** | `5` |
| **Number of Evaluated Models** | `6` |
| **Test Set Sample Size** | `1440` |
| **Statistically Significant ($lpha=0.05$)** | **YES** |

> [!NOTE]
> The p-value is below 0.05, confirming statistically significant performance differences among the models across the chronological test set.

---

## 3. Pairwise Model Significance (Wilcoxon Signed-Rank Test)

The **Wilcoxon Signed-Rank Test** performs non-parametric pairwise comparisons on absolute prediction errors ($|y - \hat{y}|$) across models.

| Model A | Model B | Statistic | p-value | Significant ($lpha=0.05$) | Superior Model |
| :--- | :--- | :---: | :---: | :---: | :--- |
| lightgbm_quantile | lstm | 12928.00 | 2.200429e-225 | YES | lightgbm_quantile |
| lightgbm_quantile | mlp | 97767.00 | 9.314417e-157 | YES | lightgbm_quantile |
| lightgbm_quantile | random_forest | 440824.00 | 7.889205e-07 | YES | lightgbm_quantile |
| lightgbm_quantile | svr | 34962.00 | 2.349259e-206 | YES | lightgbm_quantile |
| lightgbm_quantile | xgboost | 237189.00 | 3.421994e-71 | YES | lightgbm_quantile |
| lstm | mlp | 100359.00 | 7.387912e-155 | YES | mlp |
| lstm | random_forest | 59425.00 | 3.209894e-186 | YES | random_forest |
| lstm | svr | 477280.00 | 8.583636e-03 | YES | lstm |
| lstm | xgboost | 34994.00 | 2.500063e-206 | YES | xgboost |
| mlp | random_forest | 207364.00 | 1.184550e-86 | YES | random_forest |
| mlp | svr | 93247.00 | 4.256030e-160 | YES | mlp |
| mlp | xgboost | 263441.00 | 7.301774e-59 | YES | xgboost |
| random_forest | svr | 65857.00 | 4.242795e-181 | YES | random_forest |
| random_forest | xgboost | 354047.00 | 1.691916e-25 | YES | random_forest |
| svr | xgboost | 59659.00 | 4.943840e-186 | YES | xgboost |

---

## 4. Official Model Performance Ranking

Models are ranked strictly by **RMSE (ascending)** as the primary criterion, followed by **MAE** and **WMAPE**.

| Rank | Model | RMSE | MAE | WMAPE (%) | R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | lightgbm_quantile | 52.295281 | 19.326147 | 3.26% | 0.986799 |
| 2 | xgboost | 66.922507 | 36.673402 | 6.18% | 0.978382 |
| 3 | random_forest | 74.195468 | 33.236503 | 5.60% | 0.973428 |
| 4 | mlp | 89.483026 | 61.088810 | 10.30% | 0.961350 |
| 5 | lstm | 280.415240 | 208.481620 | 35.15% | 0.620449 |
| 6 | svr | 314.195321 | 227.747367 | 38.39% | 0.523497 |

---

## 5. Methodological Notes

1. **Non-Parametric Assumptions**: Forecast error distributions in supply chain demand are non-Gaussian due to monsoon seasonality and site disruptions. Therefore, non-parametric Wilcoxon and Friedman tests are mandated.
2. **Error Metric Definition**: Absolute Error $e_{i,m} = |y_i - \hat{y}_{i,m}|$ is calculated per time-series sample $i$ for model $m$.
3. **Downstream Integration**: The rank-1 model is selected as `best_model.json` and will be forwarded to **Module 4 (SHAP Explainability)** and **Module 5 (Multi-Objective Optimization)**.
