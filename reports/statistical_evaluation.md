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
| **Friedman $\chi^2$ Statistic** | `1732.0492` |
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
| lightgbm_quantile | lstm | 69698.00 | 4.482301e-178 | YES | lightgbm_quantile |
| lightgbm_quantile | mlp | 479570.00 | 1.302407e-02 | YES | mlp |
| lightgbm_quantile | random_forest | 458305.00 | 1.278945e-04 | YES | lightgbm_quantile |
| lightgbm_quantile | svr | 495676.00 | 1.435711e-01 | NO | No Statistically Significant Difference |
| lightgbm_quantile | xgboost | 515543.00 | 8.384850e-01 | NO | No Statistically Significant Difference |
| lstm | mlp | 73106.00 | 2.055693e-175 | YES | mlp |
| lstm | random_forest | 78654.00 | 4.002893e-171 | YES | random_forest |
| lstm | svr | 62726.00 | 1.392348e-183 | YES | svr |
| lstm | xgboost | 61647.00 | 1.922177e-184 | YES | xgboost |
| mlp | random_forest | 500556.00 | 2.487379e-01 | NO | No Statistically Significant Difference |
| mlp | svr | 506536.00 | 4.386218e-01 | NO | No Statistically Significant Difference |
| mlp | xgboost | 450562.00 | 1.552668e-05 | YES | xgboost |
| random_forest | svr | 456231.00 | 7.435930e-05 | YES | random_forest |
| random_forest | xgboost | 427323.00 | 6.892506e-09 | YES | xgboost |
| svr | xgboost | 496973.00 | 1.674506e-01 | NO | No Statistically Significant Difference |

---

## 4. Official Model Performance Ranking

Models are ranked strictly by **RMSE (ascending)** as the primary criterion, followed by **MAE** and **WMAPE**.

| Rank | Model | RMSE | MAE | WMAPE (%) | R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | xgboost | 97.268501 | 65.992950 | 8.47% | 0.983523 |
| 2 | mlp | 110.635444 | 73.903709 | 9.48% | 0.978683 |
| 3 | random_forest | 116.865052 | 76.971371 | 9.88% | 0.976215 |
| 4 | lightgbm_quantile | 124.534498 | 75.206677 | 9.65% | 0.972990 |
| 5 | svr | 241.176333 | 99.493466 | 12.76% | 0.898700 |
| 6 | lstm | 537.788679 | 337.150960 | 43.26% | 0.496311 |

---

## 5. Methodological Notes

1. **Non-Parametric Assumptions**: Forecast error distributions in supply chain demand are non-Gaussian due to monsoon seasonality and site disruptions. Therefore, non-parametric Wilcoxon and Friedman tests are mandated.
2. **Error Metric Definition**: Absolute Error $e_{i,m} = |y_i - \hat{y}_{i,m}|$ is calculated per time-series sample $i$ for model $m$.
3. **Downstream Integration**: The rank-1 model is selected as `best_model.json` and will be forwarded to **Module 4 (SHAP Explainability)** and **Module 5 (Multi-Objective Optimization)**.
