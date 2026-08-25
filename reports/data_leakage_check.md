# 🛡️ Data Leakage & Statistical Independence Audit Report

**Date**: 2026-08-24  
**Dataset Path**: `data/processed/processed_dataset.csv`  
**Dataset Shape**: 9,360 rows × 33 columns  
**Statistical Leakage Threshold**: $|r| > 0.95$ (Pearson correlation with target `Quantity_Required`)

---

## 1. Target Column Definition

- **Target Column Name**: `Quantity_Required`
- **Target Data Type**: `float64`
- **Summary Statistics**:
  - **Count**: 9,360
  - **Mean**: 704.22
  - **Std**: 717.13
  - **Min**: 0.00
  - **25%**: 157.00
  - **50%**: 496.00
  - **75%**: 1046.00
  - **Max**: 3716.00

---

## 2. Statistical Correlation Leakage Audit ($|r| > 0.95$)

This automated statistical audit computes Pearson correlation $r$ between every model input feature and the prediction target `Quantity_Required`. Any feature exhibiting $|r| > 0.95$ is flagged as a potential target proxy / leakage bug.

| Input Feature Name | Pearson Correlation ($r$) | Absolute Correlation ($|r|$) | Verification Result |
| :--- | :---: | :---: | :--- |
| `Rolling_Max_3` | 0.7972 | 0.7972 | ✅ PASSED |
| `EMD_IMF_6` | 0.7627 | 0.7627 | ✅ PASSED |
| `Rolling_STD_3` | 0.5891 | 0.5891 | ✅ PASSED |
| `Classical_Residual` | 0.5776 | 0.5776 | ✅ PASSED |
| `Current_Inventory` | 0.5607 | 0.5607 | ✅ PASSED |
| `Supplier` | -0.5249 | 0.5249 | ✅ PASSED |
| `Residual_Mean` | 0.4004 | 0.4004 | ✅ PASSED |
| `EMD_IMF_1` | 0.3685 | 0.3685 | ✅ PASSED |
| `Signal_Entropy` | 0.3320 | 0.3320 | ✅ PASSED |
| `EMD_IMF_5_Mean` | -0.3315 | 0.3315 | ✅ PASSED |
| `Trend_Slope` | 0.3146 | 0.3146 | ✅ PASSED |
| `Classical_Seasonal` | 0.2984 | 0.2984 | ✅ PASSED |
| `Project_ID` | 0.2939 | 0.2939 | ✅ PASSED |
| `DWT_cD_1_Entropy` | 0.2849 | 0.2849 | ✅ PASSED |
| `Region` | 0.2693 | 0.2693 | ✅ PASSED |
| `Material_Type` | -0.2654 | 0.2654 | ✅ PASSED |
| `DWT_cD_2_Entropy` | 0.2609 | 0.2609 | ✅ PASSED |
| `Seasonality_Strength` | -0.2591 | 0.2591 | ✅ PASSED |
| `DWT_cD_1_Mean` | -0.2546 | 0.2546 | ✅ PASSED |
| `EMD_IMF_1_Mean` | -0.2216 | 0.2216 | ✅ PASSED |
| `EMD_IMF_4_Mean` | -0.1897 | 0.1897 | ✅ PASSED |
| `DWT_cD_2_Mean` | 0.1600 | 0.1600 | ✅ PASSED |
| `Storage_Capacity` | 0.1185 | 0.1185 | ✅ PASSED |
| `Seasonal_Demand_Index` | 0.1178 | 0.1178 | ✅ PASSED |
| `DWT_cD_3_Entropy` | 0.1156 | 0.1156 | ✅ PASSED |
| `DWT_cA_Entropy` | 0.1020 | 0.1020 | ✅ PASSED |
| `Production_Capacity` | 0.0703 | 0.0703 | ✅ PASSED |
| `Trend_Strength` | -0.0525 | 0.0525 | ✅ PASSED |
| `DWT_cD_3_Mean` | 0.0519 | 0.0519 | ✅ PASSED |
| `EMD_IMF_3_Mean` | 0.0225 | 0.0225 | ✅ PASSED |
| `EMD_IMF_2_Mean` | -0.0012 | 0.0012 | ✅ PASSED |

---

## 3. Schema & Column Name Audit

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

---

## 4. Final Verification Verdict

### 🔒 **Audit Conclusion: PASSED**

Zero schema leaks or statistical target leakages detected. All feature correlations remain safely below the strict statistical leakage threshold ($|r| \le 0.95$). The model input feature matrix consists strictly of authentic historical lags, physical facility capacities, signal decomposition components (DWT/EMD), and categorical encoders.
