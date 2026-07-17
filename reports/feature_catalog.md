# POWERGRID Dynamic Feature Catalog

This catalog documents every feature column generated during the preprocessing (Module 1) and time-series decomposition (Module 2) stages of the pipeline, noting its origin, formula, and whether it was selected for downstream Machine Learning Forecasting (Module 3).

---

## 📋 Feature Definitions Table

| Feature Name | Description | Formula / Transform | Origin | Module | Used in Forecasting? |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Classical_Residual** | High-frequency irregular remainder component from demand signal. | `Original - Trend - Seasonal` | Quantity_Required | Module 2 | ✔ Yes |
| **Classical_Seasonal** | Additive periodic seasonal component extracted from demand signal. | `Average of seasonal detrended signals` | Quantity_Required | Module 2 | ✔ Yes |
| **Classical_Trend** | Low-frequency trend component extracted from demand signal. | `Moving Average filter (statsmodels)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_Approximation_Energy_Ratio** | Statistical signal feature representing dwt approximation energy ratio of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_Detail_Energy_Ratio** | Statistical signal feature representing dwt detail energy ratio of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Energy** | Energy (sum of squares) of approximation coefficients cA. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Entropy** | Shannon entropy of approximation coefficients cA. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cA_Mean** | Approximation wavelet coefficients representing low-frequency grid demand profile. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Std** | Approximation wavelet coefficients representing low-frequency grid demand profile. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Var** | Approximation wavelet coefficients representing low-frequency grid demand profile. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_1_Energy** | Sum of squares (energy) of detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_1_Entropy** | Shannon entropy of detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_1_Mean** | Detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_1_Std** | Detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_1_Var** | Detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_2_Energy** | Sum of squares (energy) of detail wavelet coefficients at level 2. | `Discrete Wavelet Transform Filter (Level 2)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_2_Entropy** | Shannon entropy of detail wavelet coefficients at level 2. | `Discrete Wavelet Transform Filter (Level 2)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_2_Mean** | Detail wavelet coefficients at level 2. | `Discrete Wavelet Transform Filter (Level 2)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_2_Std** | Detail wavelet coefficients at level 2. | `Discrete Wavelet Transform Filter (Level 2)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_2_Var** | Detail wavelet coefficients at level 2. | `Discrete Wavelet Transform Filter (Level 2)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_3_Energy** | Sum of squares (energy) of detail wavelet coefficients at level 3. | `Discrete Wavelet Transform Filter (Level 3)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_3_Entropy** | Shannon entropy of detail wavelet coefficients at level 3. | `Discrete Wavelet Transform Filter (Level 3)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_3_Mean** | Detail wavelet coefficients at level 3. | `Discrete Wavelet Transform Filter (Level 3)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_3_Std** | Detail wavelet coefficients at level 3. | `Discrete Wavelet Transform Filter (Level 3)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_3_Var** | Detail wavelet coefficients at level 3. | `Discrete Wavelet Transform Filter (Level 3)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Date** | Weekly timestamp of the record. | `ISO-8601 Date` | Raw Data | Module 1 | ✔ Yes |
| **Dominant_Frequency** | Statistical signal feature representing dominant frequency of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Dominant_IMF** | Statistical signal feature representing dominant imf of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_1** | Intrinsic Mode Function (IMF) #1 representing oscillatory mode of demand variation. | `EMD Sifting Mode #1` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_1_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_1_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_1_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_1_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_2** | Intrinsic Mode Function (IMF) #2 representing oscillatory mode of demand variation. | `EMD Sifting Mode #2` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_2_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_2_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_2_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_2_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_3** | Intrinsic Mode Function (IMF) #3 representing oscillatory mode of demand variation. | `EMD Sifting Mode #3` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_3_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_3_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_3_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_3_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_4** | Intrinsic Mode Function (IMF) #4 representing oscillatory mode of demand variation. | `EMD Sifting Mode #4` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_4_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_4_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_4_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_4_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_5** | Intrinsic Mode Function (IMF) #5 representing oscillatory mode of demand variation. | `EMD Sifting Mode #5` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_5_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_5_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_5_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_5_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_6** | Intrinsic Mode Function (IMF) #6 representing oscillatory mode of demand variation. | `EMD Sifting Mode #6` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_6_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_6_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_6_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_6_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Energy** | Statistical metric (Energy) calculated on the EMD Residual monotonic trend. | `np.energy(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Mean** | Statistical metric (Mean) calculated on the EMD Residual monotonic trend. | `np.mean(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Slope** | Statistical metric (Slope) calculated on the EMD Residual monotonic trend. | `np.slope(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Std** | Statistical metric (Std) calculated on the EMD Residual monotonic trend. | `np.std(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Var** | Statistical metric (Var) calculated on the EMD Residual monotonic trend. | `np.var(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Residual** | The final monotonic residual trend extracted via EMD sifting. | `Original - Sum(IMFs)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Material_Type** | Material category (e.g. Conductor, Insulator, Tower Member). | `Categorical` | Raw Data | Module 1 | ✔ Yes |
| **Num_IMFs** | Statistical signal feature representing num imfs of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Quantity_Required** | Target variable: actual material demand requested. | `Demand quantity` | Raw Data | Module 1 | ✔ Yes |
| **Region** | Geographical region code (e.g. NR, ER, WR, SR, NER). | `Categorical` | Raw Data | Module 1 | ✔ Yes |
| **Residual_Mean** | Statistical signal feature representing residual mean of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ✔ Yes |
| **Residual_Std** | Statistical signal feature representing residual std of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Residual_Variance** | Statistical signal feature representing residual variance of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Seasonality_Strength** | Statistical signal feature representing seasonality strength of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Signal_Energy** | Statistical signal feature representing signal energy of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Signal_Entropy** | Statistical signal feature representing signal entropy of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Signal_Length** | Statistical signal feature representing signal length of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Signal_Mean_Energy** | Statistical signal feature representing signal mean energy of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Trend_Slope** | Statistical signal feature representing trend slope of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Trend_Strength** | Statistical signal feature representing trend strength of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ✔ Yes |
