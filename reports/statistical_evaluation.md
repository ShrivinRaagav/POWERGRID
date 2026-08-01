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
| **Friedman $\chi^2$ Statistic** | `710.2302` |
| **p-value** | `3.014611e-151` |
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
| lightgbm_quantile | lstm | 5734.00 | 4.271435e-70 | YES | lightgbm_quantile |
| lightgbm_quantile | mlp | 12157.00 | 8.647000e-56 | YES | lightgbm_quantile |
| lightgbm_quantile | random_forest | 46939.00 | 3.352284e-07 | YES | random_forest |
| lightgbm_quantile | svr | 48840.00 | 6.144913e-06 | YES | lightgbm_quantile |
| lightgbm_quantile | xgboost | 38616.00 | 2.059661e-14 | YES | xgboost |
| lstm | mlp | 62560.00 | 7.435909e-01 | NO | No Statistically Significant Difference |
| lstm | random_forest | 5367.00 | 5.790911e-71 | YES | random_forest |
| lstm | svr | 11296.00 | 1.306011e-57 | YES | svr |
| lstm | xgboost | 2355.00 | 2.718583e-78 | YES | xgboost |
| mlp | random_forest | 21576.00 | 7.957383e-38 | YES | random_forest |
| mlp | svr | 29655.00 | 2.864091e-25 | YES | svr |
| mlp | xgboost | 15002.00 | 5.503311e-50 | YES | xgboost |
| random_forest | svr | 45478.00 | 2.871227e-08 | YES | random_forest |
| random_forest | xgboost | 57222.00 | 5.011957e-02 | NO | No Statistically Significant Difference |
| svr | xgboost | 40262.00 | 9.087968e-13 | YES | xgboost |

---

## 4. Official Model Performance Ranking

Models are ranked strictly by **RMSE (ascending)** as the primary criterion, followed by **MAE** and **WMAPE**.

| Rank | Model | RMSE | MAE | WMAPE (%) | R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | xgboost | 217.524370 | 148.924324 | 27.43% | 0.712465 |
| 2 | lightgbm_quantile | 243.542956 | 174.127621 | 32.07% | 0.639565 |
| 3 | random_forest | 265.158862 | 163.818430 | 30.17% | 0.572744 |
| 4 | svr | 296.245494 | 212.771794 | 39.19% | 0.466691 |
| 5 | mlp | 530.928348 | 416.371869 | 76.68% | -0.712961 |
| 6 | lstm | 540.141367 | 425.307788 | 78.33% | -0.772925 |

---

## 5. Methodological Notes

1. **Non-Parametric Assumptions**: Forecast error distributions in supply chain demand are non-Gaussian due to monsoon seasonality and site disruptions. Therefore, non-parametric Wilcoxon and Friedman tests are mandated.
2. **Error Metric Definition**: Absolute Error $e_{i,m} = |y_i - \hat{y}_{i,m}|$ is calculated per time-series sample $i$ for model $m$.
3. **Downstream Integration**: The rank-1 model is selected as `best_model.json` and will be forwarded to **Module 4 (SHAP Explainability)** and **Module 5 (Multi-Objective Optimization)**.
