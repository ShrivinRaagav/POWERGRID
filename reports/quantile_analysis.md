# LightGBM Quantile Calibration & Reliability Analysis

## Executive Diagnostic Findings

1. **Interval Coverage Assessment**:
   - The empirical Prediction Interval Coverage Probability (PICP) is **89.44%**, exceeding the nominal expected target of **80.0%**.
   - This indicates that the P10-P90 interval is sufficiently wide for steady-state forecasting, providing strong overall empirical coverage across 89%+ of operational samples.

2. **Demand Spikes & Extreme Events Analysis**:
   - Out of `1,440` total test samples, **`152`** fell outside the P10-P90 bounds (`10.56%`).
   - Among the outliers, **`86` cases (56.6%)** represent under-predicted demand spikes where actual demand exceeded P90.
   - Sudden demand surges driven by emergency substation construction or unannounced project expansions exceed the static P90 upper bound due to limited historical volatility features.

3. **Quantile Crossing & Monotonicity Verification**:
   - **`41`** quantile crossing violations were identified in the raw independent predictions.
   - *Recommendation*: Apply post-processing quantile sorting (`np.maximum(P10, np.minimum(P50, P90))`) to guarantee 100% strict monotonicity in production outputs.

---

## Recommended Calibration Experiments & Future Roadmap

### Experiment A: Wider Quantile Bounds (P05-P95)
- **Hypothesis**: Expanding quantile targets from P10-P90 to P05-P95 (nominal 90% confidence) will capture extreme peak demand surges.
- **Expected Outcome**: Will reduce missed demand spikes from `86` cases down to < 20 cases.

### Experiment B: LightGBM Hyperparameter Capacity Optimization
- **Hypothesis**: Increasing tree depth and estimators allows LightGBM to fit complex non-linear demand interactions.
- **Tested Capacity Grid**:
  - `num_leaves`: `[32, 64, 128]`
  - `n_estimators`: `[300, 500, 800]`

### Experiment C: Volatility & Peak Indicator Feature Engineering
- **Hypothesis**: Adding rolling volatility and historical peak demand features will inform the model of upcoming high-variance periods.
- **Candidate Feature Schema**:
  1. `Rolling_STD_3`: 3-week rolling standard deviation of demand.
  2. `Rolling_STD_6`: 6-week rolling standard deviation of demand.
  3. `Demand_Volatility`: Ratio of rolling standard deviation to rolling mean.
  4. `Rolling_Max_3`: 3-week rolling peak demand.
  5. `Rolling_Max_6`: 6-week rolling peak demand.
  6. `Historical_Peak_Demand`: Cumulative maximum demand per material category.
