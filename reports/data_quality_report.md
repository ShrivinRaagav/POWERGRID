# POWERGRID Supply Chain Optimization - Data Quality Report

This report analyzes the raw grid logistics material planning dataset and validates the changes applied by the data preparation pipeline.

---

## 1. Dataset Overview
The dataset models transmission line material logistics for POWERGRID projects. It spans 3 years of weekly observations (156 weeks) mapping material demands (conductors, insulators, towers, transformers, etc.) across 5 geographical regions, 10 states, 10 warehouses, and 5 specialized suppliers. It captures operational realisms such as monsoon transport constraints, supply shortages, weather-related risks, project budget cuts, project accelerations, and transit strikes.

---

## 2. Number of Records
*   **Raw Dataset Size**: 18180 rows, 22 columns.
*   **Cleaned/Preprocessed Dataset Size**: 12590 rows, 53 columns.
*   **Split Allocation**:
    *   *Training Split (70%)*: 4,199 rows
    *   *Validation Split (15%)*: 872 rows
    *   *Test Split (15%)*: 905 rows

---

## 3. Missing Values
*   **Raw Missing Count**: 684 null values across variables `Lead_Time_Days`, `Historical_Demand`, `Current_Inventory`, `Weather`, and `Supplier_Risk`.
*   **Imputation Strategy**: All null values were imputed based on the training split parameters to prevent data leakage (numerical columns imputed with training medians, categorical columns imputed with training modes).
*   **Cleaned Missing Count**: 0 null values.

---

## 4. Duplicates Removed
*   **Staged Duplicates in Raw Data**: 180 duplicate records.
*   **Deduplication Strategy**: Rows containing identical variables were removed, resulting in 0 duplicate records in the preprocessed stage.

---

## 5. Invalid Values Corrected
*   **Invalid Dates**: Detected and dropped rows containing unparseable date strings (such as "202-INVALID-DATE").
*   **Invalid Categorical Labels**: Identified out-of-bounds strings (e.g. `Region` = "XX", `Project_Phase` = "Unknown Phase"). These entries were replaced with their corresponding training mode labels (e.g., region mapped to "NR", project phase mapped to "Planning").

---

## 6. Outlier Summary
To prevent extreme numerical shocks from distorting forecast models (MLP, LSTM, SVR, etc.), outliers were capped using Interquartile Range (IQR) bounds $[Q1 - 1.5\times\text{IQR}, Q3 + 1.5\times\text{IQR}]$ fit on the training split:

- **Historical_Demand**: 330 values adjusted/capped.
- **Current_Inventory**: 275 values adjusted/capped.
- **Lead_Time_Days**: 107 values adjusted/capped.
- **Supplier_Risk**: 124 values adjusted/capped.
- **Transportation_Cost**: 95 values adjusted/capped.

---

## 7. Validation Summary
The validation test log below shows pre-cleaning (raw stage) failures and post-cleaning (cleaned stage) successes:

| Stage | Check Name | Column | Status | Violations | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| raw | Missing Values Check | Project_ID | **PASS** | 0 | Checks if column Project_ID contains null values. |
| raw | Missing Values Check | Date | **PASS** | 0 | Checks if column Date contains null values. |
| raw | Missing Values Check | Region | **PASS** | 0 | Checks if column Region contains null values. |
| raw | Missing Values Check | State | **PASS** | 0 | Checks if column State contains null values. |
| raw | Missing Values Check | Warehouse | **PASS** | 0 | Checks if column Warehouse contains null values. |
| raw | Missing Values Check | Supplier | **PASS** | 0 | Checks if column Supplier contains null values. |
| raw | Missing Values Check | Material_Type | **PASS** | 0 | Checks if column Material_Type contains null values. |
| raw | Missing Values Check | Project_Phase | **PASS** | 0 | Checks if column Project_Phase contains null values. |
| raw | Missing Values Check | Tower_Type | **PASS** | 0 | Checks if column Tower_Type contains null values. |
| raw | Missing Values Check | Substation_Type | **PASS** | 0 | Checks if column Substation_Type contains null values. |
| raw | Missing Values Check | Historical_Demand | **FAIL** | 147 | Checks if column Historical_Demand contains null values. |
| raw | Missing Values Check | Current_Inventory | **FAIL** | 93 | Checks if column Current_Inventory contains null values. |
| raw | Missing Values Check | Lead_Time_Days | **FAIL** | 148 | Checks if column Lead_Time_Days contains null values. |
| raw | Missing Values Check | Supplier_Risk | **FAIL** | 148 | Checks if column Supplier_Risk contains null values. |
| raw | Missing Values Check | Commodity_Price | **PASS** | 0 | Checks if column Commodity_Price contains null values. |
| raw | Missing Values Check | Transportation_Cost | **PASS** | 0 | Checks if column Transportation_Cost contains null values. |
| raw | Missing Values Check | Storage_Capacity | **PASS** | 0 | Checks if column Storage_Capacity contains null values. |
| raw | Missing Values Check | Production_Capacity | **PASS** | 0 | Checks if column Production_Capacity contains null values. |
| raw | Missing Values Check | Project_Budget | **PASS** | 0 | Checks if column Project_Budget contains null values. |
| raw | Missing Values Check | Weather | **FAIL** | 148 | Checks if column Weather contains null values. |
| raw | Missing Values Check | Season | **PASS** | 0 | Checks if column Season contains null values. |
| raw | Missing Values Check | Quantity_Required | **PASS** | 0 | Checks if column Quantity_Required contains null values. |
| raw | Duplicate Records Check | ALL_COLUMNS | **FAIL** | 180 | Checks for identical rows in the dataset. |
| raw | Negative Inventory Check | Current_Inventory | **FAIL** | 55 | Checks for negative values in the current inventory column. |
| raw | Negative Demand Check | Quantity_Required | **FAIL** | 54 | Checks for negative values in the quantity required (demand) column. |
| raw | Negative Historical Demand Check | Historical_Demand | **FAIL** | 54 | Checks for negative values in the historical demand column. |
| raw | Invalid Date Formats or Range | Date | **FAIL** | 72 | Checks if date format is parseable and falls between 2020 and 2030. |
| raw | Invalid Region Code | Region | **FAIL** | 54 | Checks if region matches one of the valid codes: ['NR', 'ER', 'WR', 'SR', 'NER'] |
| raw | Invalid Project Phase | Project_Phase | **FAIL** | 54 | Checks if project phase matches: ['Planning', 'Foundation', 'Tower Erection', 'Stringing', 'Testing & Commissioning'] |
| cleaned | Missing Values Check | Project_ID | **PASS** | 0 | Checks if column Project_ID contains null values. |
| cleaned | Missing Values Check | Date | **PASS** | 0 | Checks if column Date contains null values. |
| cleaned | Missing Values Check | Region | **PASS** | 0 | Checks if column Region contains null values. |
| cleaned | Missing Values Check | State | **PASS** | 0 | Checks if column State contains null values. |
| cleaned | Missing Values Check | Warehouse | **PASS** | 0 | Checks if column Warehouse contains null values. |
| cleaned | Missing Values Check | Supplier | **PASS** | 0 | Checks if column Supplier contains null values. |
| cleaned | Missing Values Check | Material_Type | **PASS** | 0 | Checks if column Material_Type contains null values. |
| cleaned | Missing Values Check | Project_Phase | **PASS** | 0 | Checks if column Project_Phase contains null values. |
| cleaned | Missing Values Check | Tower_Type | **PASS** | 0 | Checks if column Tower_Type contains null values. |
| cleaned | Missing Values Check | Substation_Type | **PASS** | 0 | Checks if column Substation_Type contains null values. |
| cleaned | Missing Values Check | Historical_Demand | **PASS** | 0 | Checks if column Historical_Demand contains null values. |
| cleaned | Missing Values Check | Current_Inventory | **PASS** | 0 | Checks if column Current_Inventory contains null values. |
| cleaned | Missing Values Check | Lead_Time_Days | **PASS** | 0 | Checks if column Lead_Time_Days contains null values. |
| cleaned | Missing Values Check | Supplier_Risk | **PASS** | 0 | Checks if column Supplier_Risk contains null values. |
| cleaned | Missing Values Check | Commodity_Price | **PASS** | 0 | Checks if column Commodity_Price contains null values. |
| cleaned | Missing Values Check | Transportation_Cost | **PASS** | 0 | Checks if column Transportation_Cost contains null values. |
| cleaned | Missing Values Check | Storage_Capacity | **PASS** | 0 | Checks if column Storage_Capacity contains null values. |
| cleaned | Missing Values Check | Production_Capacity | **PASS** | 0 | Checks if column Production_Capacity contains null values. |
| cleaned | Missing Values Check | Project_Budget | **PASS** | 0 | Checks if column Project_Budget contains null values. |
| cleaned | Missing Values Check | Weather | **PASS** | 0 | Checks if column Weather contains null values. |
| cleaned | Missing Values Check | Season | **PASS** | 0 | Checks if column Season contains null values. |
| cleaned | Missing Values Check | Quantity_Required | **PASS** | 0 | Checks if column Quantity_Required contains null values. |
| cleaned | Missing Values Check | Year | **PASS** | 0 | Checks if column Year contains null values. |
| cleaned | Missing Values Check | Month | **PASS** | 0 | Checks if column Month contains null values. |
| cleaned | Missing Values Check | WeekOfYear | **PASS** | 0 | Checks if column WeekOfYear contains null values. |
| cleaned | Missing Values Check | DayOfWeek | **PASS** | 0 | Checks if column DayOfWeek contains null values. |
| cleaned | Missing Values Check | Is_Quarter_End | **PASS** | 0 | Checks if column Is_Quarter_End contains null values. |
| cleaned | Missing Values Check | Month_Sin | **PASS** | 0 | Checks if column Month_Sin contains null values. |
| cleaned | Missing Values Check | Month_Cos | **PASS** | 0 | Checks if column Month_Cos contains null values. |
| cleaned | Missing Values Check | Week_Sin | **PASS** | 0 | Checks if column Week_Sin contains null values. |
| cleaned | Missing Values Check | Week_Cos | **PASS** | 0 | Checks if column Week_Cos contains null values. |
| cleaned | Missing Values Check | Lag_1 | **PASS** | 0 | Checks if column Lag_1 contains null values. |
| cleaned | Missing Values Check | Lag_2 | **PASS** | 0 | Checks if column Lag_2 contains null values. |
| cleaned | Missing Values Check | Lag_3 | **PASS** | 0 | Checks if column Lag_3 contains null values. |
| cleaned | Missing Values Check | Rolling_Mean_3 | **PASS** | 0 | Checks if column Rolling_Mean_3 contains null values. |
| cleaned | Missing Values Check | Rolling_Mean_6 | **PASS** | 0 | Checks if column Rolling_Mean_6 contains null values. |
| cleaned | Missing Values Check | Rolling_Max_3 | **PASS** | 0 | Checks if column Rolling_Max_3 contains null values. |
| cleaned | Missing Values Check | Rolling_Min_3 | **PASS** | 0 | Checks if column Rolling_Min_3 contains null values. |
| cleaned | Missing Values Check | Rolling_STD_3 | **PASS** | 0 | Checks if column Rolling_STD_3 contains null values. |
| cleaned | Missing Values Check | Demand_Velocity_1 | **PASS** | 0 | Checks if column Demand_Velocity_1 contains null values. |
| cleaned | Missing Values Check | Demand_Velocity_2 | **PASS** | 0 | Checks if column Demand_Velocity_2 contains null values. |
| cleaned | Missing Values Check | Demand_Acceleration | **PASS** | 0 | Checks if column Demand_Acceleration contains null values. |
| cleaned | Missing Values Check | Demand_Volatility_3 | **PASS** | 0 | Checks if column Demand_Volatility_3 contains null values. |
| cleaned | Missing Values Check | Peak_Spike_Flag | **PASS** | 0 | Checks if column Peak_Spike_Flag contains null values. |
| cleaned | Missing Values Check | Dip_Drop_Flag | **PASS** | 0 | Checks if column Dip_Drop_Flag contains null values. |
| cleaned | Missing Values Check | Inventory_Utilization | **PASS** | 0 | Checks if column Inventory_Utilization contains null values. |
| cleaned | Missing Values Check | Lead_Time_Category | **PASS** | 0 | Checks if column Lead_Time_Category contains null values. |
| cleaned | Missing Values Check | Demand_Growth | **PASS** | 0 | Checks if column Demand_Growth contains null values. |
| cleaned | Missing Values Check | Inventory_Coverage | **PASS** | 0 | Checks if column Inventory_Coverage contains null values. |
| cleaned | Missing Values Check | Budget_Utilization | **PASS** | 0 | Checks if column Budget_Utilization contains null values. |
| cleaned | Missing Values Check | Supplier_Risk_Score | **PASS** | 0 | Checks if column Supplier_Risk_Score contains null values. |
| cleaned | Missing Values Check | Seasonal_Demand_Index | **PASS** | 0 | Checks if column Seasonal_Demand_Index contains null values. |
| cleaned | Missing Values Check | Transportation_Cost_Index | **PASS** | 0 | Checks if column Transportation_Cost_Index contains null values. |
| cleaned | Duplicate Records Check | ALL_COLUMNS | **PASS** | 0 | Checks for identical rows in the dataset. |
| cleaned | Negative Inventory Check | Current_Inventory | **PASS** | 0 | Checks for negative values in the current inventory column. |
| cleaned | Negative Demand Check | Quantity_Required | **PASS** | 0 | Checks for negative values in the quantity required (demand) column. |
| cleaned | Negative Historical Demand Check | Historical_Demand | **PASS** | 0 | Checks for negative values in the historical demand column. |
| cleaned | Invalid Date Formats or Range | Date | **PASS** | 0 | Checks if date format is parseable and falls between 2020 and 2030. |
| cleaned | Invalid Region Code | Region | **PASS** | 0 | Checks if region matches one of the valid codes: ['NR', 'ER', 'WR', 'SR', 'NER'] |
| cleaned | Invalid Project Phase | Project_Phase | **PASS** | 0 | Checks if project phase matches: ['Planning', 'Foundation', 'Tower Erection', 'Stringing', 'Testing & Commissioning'] |

---

## 8. Feature Engineering Summary
The following statistics describe the engineered variables computed on the training partition:

| Feature Name | Mean | Median | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Lag_1 | 350.0560 | 221.5000 | 407.1341 | 0.0000 | 1858.0000 |
| Lag_2 | 349.1729 | 221.0000 | 405.8696 | 0.0000 | 1858.0000 |
| Lag_3 | 348.3257 | 221.0000 | 404.7518 | 0.0000 | 1858.0000 |
| Rolling_Mean_3 | 349.5484 | 251.3333 | 356.5454 | 0.0000 | 1858.0000 |
| Rolling_Mean_6 | 348.9562 | 258.5833 | 336.7360 | 0.0000 | 1858.0000 |
| Rolling_Max_3 | 495.4920 | 360.0000 | 479.2705 | 0.0000 | 1858.0000 |
| Rolling_Min_3 | 211.7983 | 112.0000 | 288.8894 | 0.0000 | 1858.0000 |
| Rolling_STD_3 | 152.0733 | 81.7088 | 184.8126 | 0.0000 | 1072.7168 |
| Demand_Velocity_1 | 0.8831 | 0.0000 | 335.7336 | -1858.0000 | 1858.0000 |
| Demand_Velocity_2 | 0.8472 | 0.0000 | 333.7294 | -1858.0000 | 1858.0000 |
| Demand_Acceleration | 0.0359 | 0.0000 | 571.9040 | -3344.0000 | 3670.0000 |
| Demand_Volatility_3 | 0.5446 | 0.4583 | 0.4229 | 0.0000 | 1.7321 |
| Peak_Spike_Flag | 0.2597 | 0.0000 | 0.4385 | 0.0000 | 1.0000 |
| Dip_Drop_Flag | 0.2803 | 0.0000 | 0.4492 | 0.0000 | 1.0000 |
| Inventory_Utilization | 0.0853 | 0.0464 | 0.1088 | 0.0000 | 0.6621 |
| Lead_Time_Category | 0.7490 | 1.0000 | 0.7316 | 0.0000 | 2.0000 |
| Demand_Growth | 0.2309 | 0.0000 | 0.8825 | -0.9995 | 2.0000 |
| Inventory_Coverage | 3.6407 | 1.0353 | 10.4620 | 0.0000 | 52.0000 |
| Budget_Utilization | 0.3312 | 0.1835 | 0.4419 | 0.0000 | 4.1852 |
| Supplier_Risk_Score | 0.1404 | 0.0931 | 0.1362 | 0.0030 | 0.9226 |
| Seasonal_Demand_Index | 1.0000 | 1.0670 | 0.1252 | 0.8031 | 1.1583 |
| Transportation_Cost_Index | 0.6116 | 0.5460 | 0.2884 | 0.2000 | 2.2671 |

---

## 9. Final Dataset Quality Assessment
The preprocessed dataset is determined to be of **research-quality** and ready for model training:
1. **Mathematical Stability**: Exploding variables in `Demand_Growth` and `Inventory_Coverage` have been stabilized and capped.
2. **Leakage Protection**: All imputation, scaling, and categorical encoding parameters were fit strictly on the training partition and applied to validation/test partitions chronologically.
3. **Completeness**: No missing values, duplicate records, negative inventories, or invalid categorical classes remain.
