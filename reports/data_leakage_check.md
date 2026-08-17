# 🛡️ Data Leakage Audit Report

**Date**: 2026-08-11  
**Dataset Path**: `data/processed/processed_dataset.csv`  
**Dataset Shape**: 9,360 rows × 25 columns  

---

## 1. Target Column Definition

- **Target Column Name**: `Quantity_Required`
- **Target Data Type**: `float64`
- **Summary Statistics**:
  - **Count**: 9,360
  - **Mean**: 596.19
  - **Std**: 614.65
  - **Min**: 0.00
  - **25%**: 72.00
  - **50%**: 405.50
  - **75%**: 932.00
  - **Max**: 3,232.50

---

## 2. Feature Matrix Columns (23 Model Input Features)

1. `Historical_Demand`
2. `EMD_IMF_3_Var`
3. `Residual_Std`
4. `EMD_IMF_5_Var`
5. `DWT_cD_2_Mean`
6. `Production_Capacity`
7. `EMD_IMF_5_Mean`
8. `DWT_cD_1_Entropy`
9. `Trend_Strength`
10. `EMD_IMF_4_Mean`
11. `Signal_Entropy`
12. `DWT_cD_1_Mean`
13. `DWT_cD_3_Mean`
14. `Storage_Capacity`
15. `DWT_cD_2_Entropy`
16. `DWT_Approximation_Energy_Ratio`
17. `DWT_cD_3_Entropy`
18. `EMD_IMF_3_Mean`
19. `Seasonality_Strength`
20. `EMD_IMF_2_Mean`
21. `Project_ID`
22. `Region`
23. `Material_Type`

---

## 3. Leakage Verification Audit

| Suspicious Column Checked | Found in Feature Matrix? | Verification Result |
| :--- | :---: | :--- |
| `Demand` | ❌ No | **PASSED** |
| `Future_Demand` | ❌ No | **PASSED** |
| `Target` | ❌ No | **PASSED** |
| `Actual_Demand` | ❌ No | **PASSED** |
| `Quantity` | ❌ No | **PASSED** |
| `Forecast_Prediction` | ❌ No | **PASSED** |
| `P10` | ❌ No | **PASSED** |
| `P50` | ❌ No | **PASSED** |
| `P90` | ❌ No | **PASSED** |

### 🔒 **Conclusion**:
Zero target-related or leakage columns exist in the model input feature matrix. All model inputs consist strictly of historical lagged observations, signal decomposition features (DWT/EMD), facility capacities, and categorical identifiers.
