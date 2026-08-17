# 📅 Chronological Dataset Split Verification Report

**Date**: 2026-08-11  
**Total Processed Rows**: 9,360  
**Splitting Strategy**: Strict Chronological Out-of-Sample Time-Series Split (70% Train / 15% Validation / 15% Test)

---

## 1. Split Size & Date Ranges

| Split Subset | Percentage | Row Count | Start Date | End Date | Purpose & Constraints |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Set** | 70% | **6,552** | `2020-01-01` | `2024-03-06` | Model training & parameter fitting |
| **Validation Set** | 15% | **1,404** | `2024-03-06` | `2025-01-29` | Hyperparameter tuning & early stopping |
| **Independent Test Set** | 15% | **1,404** | `2025-01-29` | `2025-12-17` | Hold-out final out-of-sample evaluation |

---

## 2. Integrity & Isolation Verification

- **Training Isolation**: Feature scaling (`StandardScaler`), feature selection (`FeatureSelector`), and encoders were fit **strictly on the Training Set** ($N=6,552$) and applied downstream to Validation and Test sets without data leakage.
- **Test Set Isolation**: The test set ($N=1,404$) was never used for parameter fitting, feature selection, scaling, hyperparameter tuning, or early stopping.
- **Chronological Sequence**: All time series records are ordered strictly by timestamp (`Date`), preventing future look-ahead bias.
