# LightGBM Quantile Material-Level Uncertainty Calibration Summary

## Overview

Categorizes grid material categories into specific uncertainty calibration profiles to balance stockout risk vs. inventory holding cost.

---

## 🚨 1. High Risk Under-Covered Materials

Criteria: `P90 Violations > 10%` OR `PICP Coverage < 80%`

### Conductor
- **Empirical PICP**: `82.91%` (Expected 80%)
- **P90 Violations (Missed Peaks)**: `41` samples (`17.1%` of category test set)
- **Coefficient of Variation (CV)**: `0.2443` (High volatility)
- **Operational Risk**: High risk of material shortages and transmission line construction delays during unannounced demand spikes.
- **Recommendation**: Expand quantile bounds to **P05-P95** or apply dynamic safety stock buffers in Module 5.

---

## 🛡️ 2. Over-Conservative Materials

Criteria: `PICP Coverage > 95%` AND `Interval Width > 1.5x Demand Standard Deviation`

### Tower Member
- **Empirical PICP**: `96.25%` (Over-covered)
- **Average Interval Width**: `458.29` units vs. Demand Std `256.09` units
- **Operational Risk**: Overestimating uncertainty causes excessive buffer stock allocation, leading to tied-up capital and warehouse holding cost.
- **Recommendation**: Narrow quantile bounds to **P15-P85** or reduce inventory safety buffer.

### Transformer
- **Empirical PICP**: `99.58%` (Over-covered)
- **Average Interval Width**: `200.87` units vs. Demand Std `5.02` units
- **Operational Risk**: Overestimating uncertainty causes excessive buffer stock allocation, leading to tied-up capital and warehouse holding cost.
- **Recommendation**: Narrow quantile bounds to **P15-P85** or reduce inventory safety buffer.

---

## ✅ 3. Well-Calibrated Material Categories

- **Earthwire**: PICP = `86.25%`, Avg Width = `77.28` units, P90 Violations = `13` samples.
- **Hardware Fittings**: PICP = `80.42%`, Avg Width = `126.81` units, P90 Violations = `16` samples.
- **Insulator**: PICP = `92.50%`, Avg Width = `232.16` units, P90 Violations = `13` samples.
