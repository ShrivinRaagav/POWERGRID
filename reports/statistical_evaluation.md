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
| **Friedman $\chi^2$ Statistic** | `615.7676` |
| **p-value** | `7.921884e-131` |
| **Degrees of Freedom** | `5` |
| **Number of Evaluated Models** | `6` |
| **Test Set Sample Size** | `504` |
| **Statistically Significant ($lpha=0.05$)** | **YES** |

> [!NOTE]
> The p-value is below 0.05, confirming statistically significant performance differences among the models across the chronological test set.

---

## 3. Pairwise Model Significance (Wilcoxon Signed-Rank Test)

The **Wilcoxon Signed-Rank Test** performs non-parametric pairwise comparisons on absolute prediction errors ($|y - \hat{y}|$) across models.

| Model A | Model B | Statistic | p-value | Significant ($lpha=0.05$) | Superior Model |
| :--- | :--- | :---: | :---: | :---: | :--- |
| lightgbm_quantile | lstm | 5086.00 | 1.243296e-71 | YES | lightgbm_quantile |
| lightgbm_quantile | mlp | 34064.00 | 1.589788e-19 | YES | lightgbm_quantile |
| lightgbm_quantile | random_forest | 42995.00 | 2.823495e-10 | YES | random_forest |
| lightgbm_quantile | svr | 48840.00 | 6.144913e-06 | YES | lightgbm_quantile |
| lightgbm_quantile | xgboost | 44531.00 | 5.263653e-09 | YES | xgboost |
| lstm | mlp | 40994.00 | 4.520161e-12 | YES | mlp |
| lstm | random_forest | 3942.00 | 2.194895e-74 | YES | random_forest |
| lstm | svr | 9114.00 | 2.328219e-62 | YES | svr |
| lstm | xgboost | 2123.00 | 7.155927e-79 | YES | xgboost |
| mlp | random_forest | 29897.00 | 6.202235e-25 | YES | random_forest |
| mlp | svr | 50845.00 | 9.290659e-05 | YES | svr |
| mlp | xgboost | 29830.00 | 5.010495e-25 | YES | xgboost |
| random_forest | svr | 43270.00 | 4.843152e-10 | YES | random_forest |
| random_forest | xgboost | 59756.00 | 2.362976e-01 | NO | No Statistically Significant Difference |
| svr | xgboost | 40622.00 | 2.012650e-12 | YES | xgboost |

---

## 4. Official Model Performance Ranking

Models are ranked strictly by **RMSE (ascending)** as the primary criterion, followed by **MAE** and **WMAPE**.

| Rank | Model | RMSE | MAE | WMAPE (%) | R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | xgboost | 218.515741 | 150.320743 | 27.68% | 0.709838 |
| 2 | lightgbm_quantile | 243.542956 | 174.127621 | 32.07% | 0.639565 |
| 3 | random_forest | 253.719658 | 157.834466 | 29.07% | 0.608814 |
| 4 | svr | 296.245494 | 212.771794 | 39.19% | 0.466691 |
| 5 | lstm | 550.128394 | 430.951906 | 79.37% | -0.839093 |
| 6 | mlp | 588.592233 | 352.260413 | 64.88% | -1.105255 |

---

## 5. Methodological Notes

1. **Non-Parametric Assumptions**: Forecast error distributions in supply chain demand are non-Gaussian due to monsoon seasonality and site disruptions. Therefore, non-parametric Wilcoxon and Friedman tests are mandated.
2. **Error Metric Definition**: Absolute Error $e_{i,m} = |y_i - \hat{y}_{i,m}|$ is calculated per time-series sample $i$ for model $m$.
3. **Downstream Integration**: The rank-1 model is selected as `best_model.json` and will be forwarded to **Module 4 (SHAP Explainability)** and **Module 5 (Multi-Objective Optimization)**.
