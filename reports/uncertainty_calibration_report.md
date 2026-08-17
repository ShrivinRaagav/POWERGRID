# Prediction Interval Coverage Analysis

## Overview

- **Model**: LightGBM Quantile Regression
- **Target Prediction Interval**: P10-P90 (Expected 80.0% Nominal Coverage)

---

## 1. Prediction Interval Coverage Probability (PICP)

- **Total Test Samples**: `1,440`
- **Covered Samples (P10 <= Actual <= P90)**: `1,291`
- **Outside Interval Samples**: `149`
- **Empirical Coverage (PICP)**: `89.65%`
- **Expected Nominal Coverage**: `80.00%`
- **Calibration Status**: `OVER-CONSERVATIVE (WELL COVERED)`

---

## 2. Prediction Interval Width Statistics

- **Average Interval Width**: `286.46` units
- **Minimum Interval Width**: `15.71` units
- **Maximum Interval Width**: `1,560.71` units

---

## 3. Quantile Monotonicity & Consistency Audit

- **Quantile Crossing Count (P10 > P50 or P50 > P90)**: `14` violations
- **Consistency Status**: `WARNING (14 Crossing Violations Found)`

---

## 4. Outlier Analysis Summary

- **Total Outlier Observations**: `149` points (`10.35%` of test set)
- **Under-Predicted Spikes (Actual > P90)**: `91` cases
- **Over-Predicted Dips (Actual < P10)**: `58` cases
- **Outlier Table Saved To**: `reports/forecast_interval_outliers.csv`
