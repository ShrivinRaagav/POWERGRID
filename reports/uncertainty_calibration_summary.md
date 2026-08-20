# LightGBM Quantile Material-Level Uncertainty Calibration Summary

## Overview

Categorizes grid material categories into specific uncertainty calibration profiles to balance stockout risk vs. inventory holding cost.

---

## 🚨 1. High Risk Under-Covered Materials

Criteria: `P90 Violations > 10%` OR `PICP Coverage < 80%`

### Conductor
- **Empirical PICP**: `85.83%` (Expected 80%)
- **P90 Violations (Missed Peaks)**: `34` samples (`14.2%` of category test set)
- **Coefficient of Variation (CV)**: `0.2443` (High volatility)
- **Operational Risk**: High risk of material shortages and transmission line construction delays during unannounced demand spikes.
- **Recommendation**: Expand quantile bounds to **P05-P95** or apply dynamic safety stock buffers in Module 5.

---

## 🛡️ 2. Over-Conservative Materials

Criteria: `PICP Coverage > 95%` AND `Interval Width > 1.5x Demand Standard Deviation`

### Transformer
- **Empirical PICP**: `99.58%` (Over-covered)
- **Average Interval Width**: `354.93` units vs. Demand Std `5.02` units
- **Operational Risk**: Overestimating uncertainty causes excessive buffer stock allocation, leading to tied-up capital and warehouse holding cost.
- **Recommendation**: Narrow quantile bounds to **P15-P85** or reduce inventory safety buffer.

---

## ✅ 3. Well-Calibrated Material Categories

- **Earthwire**: PICP = `87.08%`, Avg Width = `144.08` units, P90 Violations = `2` samples.
- **Hardware Fittings**: PICP = `84.17%`, Avg Width = `168.16` units, P90 Violations = `11` samples.
- **Insulator**: PICP = `88.75%`, Avg Width = `240.85` units, P90 Violations = `23` samples.
- **Tower Member**: PICP = `91.25%`, Avg Width = `487.76` units, P90 Violations = `16` samples.
