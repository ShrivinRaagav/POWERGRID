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

- **Historical_Demand**: 658 values adjusted/capped.
- **Current_Inventory**: 1032 values adjusted/capped.
- **Lead_Time_Days**: 214 values adjusted/capped.
- **Supplier_Risk**: 658 values adjusted/capped.
- **Commodity_Price**: 1156 values adjusted/capped.
- **Transportation_Cost**: 586 values adjusted/capped.
- **Project_Budget**: 98 values adjusted/capped.

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
| Lag_1 | 280.3025 | 160.0000 | 312.6494 | 0.0000 | 1077.5000 |
| Lag_2 | 278.4149 | 160.0000 | 311.0103 | 0.0000 | 1077.5000 |
| Lag_3 | 276.5593 | 160.0000 | 309.2987 | 0.0000 | 1077.5000 |
| Rolling_Mean_3 | 277.9092 | 166.6667 | 300.2694 | 0.0000 | 1077.5000 |
| Rolling_Mean_6 | 274.4063 | 166.0000 | 294.0213 | 0.0000 | 1077.5000 |
| Rolling_Max_3 | 333.4632 | 206.0000 | 343.3980 | 0.0000 | 1077.5000 |
| Rolling_Min_3 | 226.0838 | 133.0000 | 271.5477 | 0.0000 | 1077.5000 |
| Rolling_STD_3 | 57.8938 | 24.2040 | 82.2543 | 0.0000 | 622.0949 |
| Demand_Velocity_1 | 1.8876 | 0.0000 | 140.1286 | -1077.5000 | 1077.5000 |
| Demand_Velocity_2 | 1.8556 | 0.0000 | 139.8670 | -1077.5000 | 1077.5000 |
| Demand_Acceleration | 0.0320 | 0.0000 | 238.0325 | -1496.0000 | 2155.0000 |
| Demand_Volatility_3 | 0.3222 | 0.2467 | 0.3363 | 0.0000 | 1.7321 |
| Peak_Spike_Flag | 0.1643 | 0.0000 | 0.3705 | 0.0000 | 1.0000 |
| Dip_Drop_Flag | 0.1649 | 0.0000 | 0.3711 | 0.0000 | 1.0000 |
| Inventory_Utilization | 0.0847 | 0.0433 | 0.1026 | 0.0000 | 0.5303 |
| Lead_Time_Category | 0.8276 | 1.0000 | 0.7393 | 0.0000 | 2.0000 |
| Demand_Growth | 0.1257 | 0.0000 | 0.6219 | -0.9991 | 2.0000 |
| Inventory_Coverage | 7.0523 | 1.0385 | 14.7634 | 0.0000 | 52.0000 |
| Budget_Utilization | 0.2717 | 0.1389 | 0.3569 | 0.0000 | 2.7780 |
| Supplier_Risk_Score | 0.1489 | 0.1038 | 0.1295 | 0.0030 | 0.6283 |
| Seasonal_Demand_Index | 1.0000 | 0.9631 | 0.1649 | 0.7943 | 1.3827 |
| Transportation_Cost_Index | 0.5561 | 0.4951 | 0.2467 | 0.2000 | 1.4619 |

---

## 9. Final Dataset Quality Assessment
The preprocessed dataset is determined to be of **research-quality** and ready for model training:
1. **Mathematical Stability**: Exploding variables in `Demand_Growth` and `Inventory_Coverage` have been stabilized and capped.
2. **Leakage Protection**: All imputation, scaling, and categorical encoding parameters were fit strictly on the training partition and applied to validation/test partitions chronologically.
3. **Completeness**: No missing values, duplicate records, negative inventories, or invalid categorical classes remain.
