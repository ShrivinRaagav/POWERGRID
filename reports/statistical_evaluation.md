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
| **Friedman $\chi^2$ Statistic** | `2348.1254` |
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
| lightgbm_quantile | lstm | 47173.00 | 3.565455e-196 | YES | lightgbm_quantile |
| lightgbm_quantile | mlp | 274749.50 | 6.389520e-54 | YES | lightgbm_quantile |
| lightgbm_quantile | random_forest | 417248.00 | 1.260322e-10 | YES | lightgbm_quantile |
| lightgbm_quantile | svr | 35734.50 | 1.053538e-205 | YES | lightgbm_quantile |
| lightgbm_quantile | xgboost | 360709.50 | 1.320023e-23 | YES | lightgbm_quantile |
| lstm | mlp | 86194.00 | 2.218090e-165 | YES | mlp |
| lstm | random_forest | 68174.00 | 2.849692e-179 | YES | random_forest |
| lstm | svr | 291030.00 | 3.391947e-47 | YES | lstm |
| lstm | xgboost | 71766.00 | 1.857402e-176 | YES | xgboost |
| mlp | random_forest | 360356.50 | 1.052641e-23 | YES | random_forest |
| mlp | svr | 56884.00 | 2.907160e-188 | YES | mlp |
| mlp | xgboost | 405519.00 | 7.228427e-13 | YES | xgboost |
| random_forest | svr | 48192.00 | 2.454493e-195 | YES | random_forest |
| random_forest | xgboost | 454679.00 | 4.902548e-05 | YES | xgboost |
| svr | xgboost | 51965.00 | 2.996200e-192 | YES | xgboost |

---

## 4. Official Model Performance Ranking

Models are ranked strictly by **RMSE (ascending)** as the primary criterion, followed by **MAE** and **WMAPE**.

| Rank | Model | RMSE | MAE | WMAPE (%) | R² |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | lightgbm_quantile | 50.601806 | 17.837354 | 3.01% | 0.987641 |
| 2 | xgboost | 53.818874 | 22.861951 | 3.85% | 0.986019 |
| 3 | mlp | 56.929308 | 27.111421 | 4.57% | 0.984356 |
| 4 | random_forest | 57.209336 | 23.356351 | 3.94% | 0.984202 |
| 5 | lstm | 228.924052 | 146.528785 | 24.70% | 0.747041 |
| 6 | svr | 313.886858 | 221.705513 | 37.38% | 0.524432 |

---

## 5. Methodological Notes

1. **Non-Parametric Assumptions**: Forecast error distributions in supply chain demand are non-Gaussian due to monsoon seasonality and site disruptions. Therefore, non-parametric Wilcoxon and Friedman tests are mandated.
2. **Error Metric Definition**: Absolute Error $e_{i,m} = |y_i - \hat{y}_{i,m}|$ is calculated per time-series sample $i$ for model $m$.
3. **Downstream Integration**: The rank-1 model is selected as `best_model.json` and will be forwarded to **Module 4 (SHAP Explainability)** and **Module 5 (Multi-Objective Optimization)**.
