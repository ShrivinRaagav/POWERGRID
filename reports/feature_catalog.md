# POWERGRID Dynamic Feature Catalog

This catalog documents every feature column generated during the preprocessing (Module 1) and time-series decomposition (Module 2) stages of the pipeline, noting its origin, formula, and whether it was selected for downstream Machine Learning Forecasting (Module 3).

---

## 📋 Feature Definitions Table

| Feature Name | Description | Formula / Transform | Origin | Module | Used in Forecasting? |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Budget_Utilization** | Remaining financial budget allocation metric. | `Project_Budget / Capacity` | Project_Budget | Module 1 | ❌ No (Dropped) |
| **Classical_Residual** | High-frequency irregular remainder component from demand signal. | `Original - Trend - Seasonal` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Classical_Seasonal** | Additive periodic seasonal component extracted from demand signal. | `Average of seasonal detrended signals` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Classical_Trend** | Low-frequency trend component extracted from demand signal. | `Moving Average filter (statsmodels)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Commodity_Price** | Market raw material price index. | `Market Index` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Current_Inventory** | Currently available stock level at the warehouse. | `Stock count` | Raw Data | Module 1 | ❌ No (Dropped) |
| **DWT_Approximation_Energy_Ratio** | Statistical signal feature representing dwt approximation energy ratio of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_Detail_Energy_Ratio** | Statistical signal feature representing dwt detail energy ratio of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Energy** | Energy (sum of squares) of approximation coefficients cA. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Entropy** | Shannon entropy of approximation coefficients cA. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Mean** | Approximation wavelet coefficients representing low-frequency grid demand profile. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Std** | Approximation wavelet coefficients representing low-frequency grid demand profile. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cA_Var** | Approximation wavelet coefficients representing low-frequency grid demand profile. | `DWT Low-Pass Filter` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_1_Energy** | Sum of squares (energy) of detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_1_Entropy** | Shannon entropy of detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_1_Mean** | Detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ✔ Yes |
| **DWT_cD_1_Std** | Detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_1_Var** | Detail wavelet coefficients at level 1. | `Discrete Wavelet Transform Filter (Level 1)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **DWT_cD_2_Energy** | Sum of squares (energy) of detail wavelet coefficients at level 2. | `Discrete Wavelet Transform Filter (Level 2)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
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
| **DayOfWeek** | Day index of the week (0 to 6) from Date. | `dt.dayofweek` | Date Column | Module 1 | ❌ No (Dropped) |
| **Demand_Growth** | Demand growth speed relative to rolling average. | `(Lag_1 - Rolling_Mean_3) / (Rolling_Mean_3 + epsilon)` | Quantity_Required | Module 1 | ❌ No (Dropped) |
| **Dominant_Frequency** | Statistical signal feature representing dominant frequency of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Dominant_IMF** | Statistical signal feature representing dominant imf of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_1** | Intrinsic Mode Function (IMF) #1 representing oscillatory mode of demand variation. | `EMD Sifting Mode #1` | Quantity_Required | Module 2 | ❌ No (Dropped) |
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
| **EMD_IMF_3_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_3_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_3_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_4** | Intrinsic Mode Function (IMF) #4 representing oscillatory mode of demand variation. | `EMD Sifting Mode #4` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_4_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_4_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_4_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_4_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_5** | Intrinsic Mode Function (IMF) #5 representing oscillatory mode of demand variation. | `EMD Sifting Mode #5` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_5_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_5_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_5_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_5_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ✔ Yes |
| **EMD_IMF_6** | Intrinsic Mode Function (IMF) #6 representing oscillatory mode of demand variation. | `EMD Sifting Mode #6` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_6_Energy** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_6_Mean** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_6_Std** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_IMF_6_Var** | Intrinsic Mode Function (IMF) #? representing oscillatory mode of demand variation. | `EMD Sifting Mode #?` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Energy** | Statistical metric (Energy) calculated on the EMD Residual monotonic trend. | `np.energy(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Mean** | Statistical metric (Mean) calculated on the EMD Residual monotonic trend. | `np.mean(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Slope** | Statistical metric (Slope) calculated on the EMD Residual monotonic trend. | `np.slope(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Std** | Statistical metric (Std) calculated on the EMD Residual monotonic trend. | `np.std(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Resid_Var** | Statistical metric (Var) calculated on the EMD Residual monotonic trend. | `np.var(EMD_Residual)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **EMD_Residual** | The final monotonic residual trend extracted via EMD sifting. | `Original - Sum(IMFs)` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Historical_Demand** | Last month's historical demand baseline. | `Numeric Value` | Raw Data | Module 1 | ✔ Yes |
| **Inventory_Coverage** | Warehouse stock replenishment safety margin in weeks. | `Current_Inventory / (Rolling_Mean_3 + epsilon)` | Inventory, Lags | Module 1 | ❌ No (Dropped) |
| **Inventory_Utilization** | Inventory fill-rate ratio of warehouse capacity. | `Current_Inventory / Storage_Capacity` | Inventory, Capacity | Module 1 | ❌ No (Dropped) |
| **Is_Quarter_End** | Flag indicating if the date lies at the end of a fiscal quarter. | `dt.is_quarter_end` | Date Column | Module 1 | ❌ No (Dropped) |
| **Lag_1** | One-period historical lag of target Quantity_Required. | `Shift(1) grouped by Warehouse & Material` | Quantity_Required | Module 1 | ❌ No (Dropped) |
| **Lag_2** | Two-period historical lag of target Quantity_Required. | `Shift(2) grouped by Warehouse & Material` | Quantity_Required | Module 1 | ❌ No (Dropped) |
| **Lag_3** | Three-period historical lag of target Quantity_Required. | `Shift(3) grouped by Warehouse & Material` | Quantity_Required | Module 1 | ❌ No (Dropped) |
| **Lead_Time_Category** | Categorized lead time duration index (0: Short, 1: Medium, 2: Long). | `Cut(Lead_Time_Days, Bins)` | Lead_Time_Days | Module 1 | ❌ No (Dropped) |
| **Lead_Time_Days** | Supply duration in days from order to delivery. | `Duration` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Material_Type** | Material category (e.g. Conductor, Insulator, Tower Member). | `Categorical` | Raw Data | Module 1 | ✔ Yes |
| **Month** | Calendar month index (1 to 12) from Date. | `dt.month` | Date Column | Module 1 | ❌ No (Dropped) |
| **Month_Cos** | Cosine transformation of month index for cyclical mapping. | `cos(2 * pi * Month / 12)` | Date Column | Module 1 | ❌ No (Dropped) |
| **Month_Sin** | Sine transformation of month index for cyclical mapping. | `sin(2 * pi * Month / 12)` | Date Column | Module 1 | ❌ No (Dropped) |
| **Num_IMFs** | Statistical signal feature representing num imfs of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Production_Capacity** | Supplier manufacturing capacity limit. | `Capacity limit` | Raw Data | Module 1 | ✔ Yes |
| **Project_Budget** | Remaining financial budget allocation for the project. | `Financial Amount` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Project_ID** | Unique identifier for the transmission line project. | `Categorical Identifier` | Raw Data | Module 1 | ✔ Yes |
| **Project_Phase** | Current phase of project construction. | `Categorical` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Quantity_Required** | Target variable: actual material demand requested. | `Demand quantity` | Raw Data | Module 1 | ✔ Yes |
| **Region** | Geographical region code (e.g. NR, ER, WR, SR, NER). | `Categorical` | Raw Data | Module 1 | ✔ Yes |
| **Residual_Mean** | Statistical signal feature representing residual mean of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Residual_Std** | Statistical signal feature representing residual std of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ✔ Yes |
| **Residual_Variance** | Statistical signal feature representing residual variance of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Rolling_Mean_3** | 3-week rolling average demand shift-1 lag sequence. | `RollingMean(3) of Lag_1` | Quantity_Required | Module 1 | ❌ No (Dropped) |
| **Rolling_Mean_6** | 6-week rolling average demand shift-1 lag sequence. | `RollingMean(6) of Lag_1` | Quantity_Required | Module 1 | ❌ No (Dropped) |
| **Season** | Climatic season (Summer, Monsoon, Winter, etc.). | `Categorical` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Seasonal_Demand_Index** | Mean demand index of material in season relative to overall average. | `Mean(Material_Season) / Mean(Material_Overall)` | Material_Type, Season, Target | Module 1 | ❌ No (Dropped) |
| **Seasonality_Strength** | Statistical signal feature representing seasonality strength of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ✔ Yes |
| **Signal_Energy** | Statistical signal feature representing signal energy of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Signal_Entropy** | Statistical signal feature representing signal entropy of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ✔ Yes |
| **Signal_Length** | Statistical signal feature representing signal length of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Signal_Mean_Energy** | Statistical signal feature representing signal mean energy of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **State** | State where the project is located. | `Categorical` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Storage_Capacity** | Warehouse storage limit. | `Capacity limit` | Raw Data | Module 1 | ✔ Yes |
| **Substation_Type** | Substation insulation method (e.g. AIS, GIS). | `Categorical` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Supplier** | Vendor providing the material. | `Categorical` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Supplier_Risk** | Assessed supplier operational risk score (0.0 to 1.0). | `Probability/Index` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Supplier_Risk_Score** | Scaled and penalized supplier reliability index. | `Supplier_Risk * Lead_Time_Days` | Supplier_Risk, Lead_Time | Module 1 | ❌ No (Dropped) |
| **Tower_Type** | Structural type of transmission tower used. | `Categorical` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Transportation_Cost** | Freight shipping tariff index. | `Cost Index` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Transportation_Cost_Index** | Transportation index scaled against supplier delay risks. | `Transportation_Cost * Supplier_Risk` | Transportation_Cost, Risk | Module 1 | ❌ No (Dropped) |
| **Trend_Slope** | Statistical signal feature representing trend slope of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ❌ No (Dropped) |
| **Trend_Strength** | Statistical signal feature representing trend strength of the demand curve. | `Signal Analysis Statistic` | Quantity_Required | Module 2 | ✔ Yes |
| **Warehouse** | Target warehouse identifier. | `Categorical` | Raw Data | Module 1 | ❌ No (Dropped) |
| **Weather** | Severe local weather conditions (Extreme Cold, Heat, Normal). | `Categorical` | Raw Data | Module 1 | ❌ No (Dropped) |
| **WeekOfYear** | Calendar week index (1 to 53) from Date. | `dt.isocalendar().week` | Date Column | Module 1 | ❌ No (Dropped) |
| **Week_Cos** | Cosine transformation of week index for cyclical mapping. | `cos(2 * pi * Week / 52.177)` | Date Column | Module 1 | ❌ No (Dropped) |
| **Week_Sin** | Sine transformation of week index for cyclical mapping. | `sin(2 * pi * Week / 52.177)` | Date Column | Module 1 | ❌ No (Dropped) |
| **Year** | Calendar year extracted from Date. | `dt.year` | Date Column | Module 1 | ❌ No (Dropped) |
