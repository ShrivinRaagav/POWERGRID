# POWERGRID Supply Chain Optimization - Data Quality Report

This report analyzes the raw grid logistics material planning dataset and validates the changes applied by the data preparation pipeline.

---

## 1. Dataset Overview
The dataset models transmission line material logistics for POWERGRID projects. It spans 3 years of weekly observations (156 weeks) mapping material demands (conductors, insulators, towers, transformers, etc.) across 5 geographical regions, 10 states, 10 warehouses, and 5 specialized suppliers. It captures operational realisms such as monsoon transport constraints, supply shortages, weather-related risks, project budget cuts, project accelerations, and transit strikes.

---

## 2. Number of Records
*   **Raw Dataset Size**: 6060 rows, 22 columns.
*   **Cleaned/Preprocessed Dataset Size**: 4199 rows, 44 columns.
*   **Split Allocation**:
    *   *Training Split (70%)*: 4,199 rows
    *   *Validation Split (15%)*: 872 rows
    *   *Test Split (15%)*: 905 rows

---

## 3. Missing Values
*   **Raw Missing Count**: 225 null values across variables `Lead_Time_Days`, `Historical_Demand`, `Current_Inventory`, `Weather`, and `Supplier_Risk`.
*   **Imputation Strategy**: All null values were imputed based on the training split parameters to prevent data leakage (numerical columns imputed with training medians, categorical columns imputed with training modes).
*   **Cleaned Missing Count**: 0 null values.

---

## 4. Duplicates Removed
*   **Staged Duplicates in Raw Data**: 60 duplicate records.
*   **Deduplication Strategy**: Rows containing identical variables were removed, resulting in 0 duplicate records in the preprocessed stage.

---

## 5. Invalid Values Corrected
*   **Invalid Dates**: Detected and dropped rows containing unparseable date strings (such as "202-INVALID-DATE").
*   **Invalid Categorical Labels**: Identified out-of-bounds strings (e.g. `Region` = "XX", `Project_Phase` = "Unknown Phase"). These entries were replaced with their corresponding training mode labels (e.g., region mapped to "NR", project phase mapped to "Planning").

---

## 6. Outlier Summary
To prevent extreme numerical shocks from distorting forecast models (MLP, LSTM, SVR, etc.), outliers were capped using Interquartile Range (IQR) bounds $[Q1 - 1.5\times\text{IQR}, Q3 + 1.5\times\text{IQR}]$ fit on the training split:

- **Historical_Demand**: 227 values adjusted/capped.
- **Current_Inventory**: 104 values adjusted/capped.
- **Lead_Time_Days**: 87 values adjusted/capped.
- **Supplier_Risk**: 293 values adjusted/capped.
- **Transportation_Cost**: 260 values adjusted/capped.

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
| raw | Missing Values Check | Historical_Demand | **FAIL** | 48 | Checks if column Historical_Demand contains null values. |
| raw | Missing Values Check | Current_Inventory | **FAIL** | 30 | Checks if column Current_Inventory contains null values. |
| raw | Missing Values Check | Lead_Time_Days | **FAIL** | 49 | Checks if column Lead_Time_Days contains null values. |
| raw | Missing Values Check | Supplier_Risk | **FAIL** | 49 | Checks if column Supplier_Risk contains null values. |
| raw | Missing Values Check | Commodity_Price | **PASS** | 0 | Checks if column Commodity_Price contains null values. |
| raw | Missing Values Check | Transportation_Cost | **PASS** | 0 | Checks if column Transportation_Cost contains null values. |
| raw | Missing Values Check | Storage_Capacity | **PASS** | 0 | Checks if column Storage_Capacity contains null values. |
| raw | Missing Values Check | Production_Capacity | **PASS** | 0 | Checks if column Production_Capacity contains null values. |
| raw | Missing Values Check | Project_Budget | **PASS** | 0 | Checks if column Project_Budget contains null values. |
| raw | Missing Values Check | Weather | **FAIL** | 49 | Checks if column Weather contains null values. |
| raw | Missing Values Check | Season | **PASS** | 0 | Checks if column Season contains null values. |
| raw | Missing Values Check | Quantity_Required | **PASS** | 0 | Checks if column Quantity_Required contains null values. |
| raw | Duplicate Records Check | ALL_COLUMNS | **FAIL** | 60 | Checks for identical rows in the dataset. |
| raw | Negative Inventory Check | Current_Inventory | **FAIL** | 19 | Checks for negative values in the current inventory column. |
| raw | Negative Demand Check | Quantity_Required | **FAIL** | 18 | Checks for negative values in the quantity required (demand) column. |
| raw | Negative Historical Demand Check | Historical_Demand | **FAIL** | 18 | Checks for negative values in the historical demand column. |
| raw | Invalid Date Formats or Range | Date | **FAIL** | 24 | Checks if date format is parseable and falls between 2020 and 2030. |
| raw | Invalid Region Code | Region | **FAIL** | 18 | Checks if region matches one of the valid codes: ['NR', 'ER', 'WR', 'SR', 'NER'] |
| raw | Invalid Project Phase | Project_Phase | **FAIL** | 18 | Checks if project phase matches: ['Planning', 'Foundation', 'Tower Erection', 'Stringing', 'Testing & Commissioning'] |
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
| Lag_1 | 281.7226 | 164.0000 | 311.5279 | 0.0000 | 1075.0000 |
| Lag_2 | 278.0069 | 164.0000 | 308.2889 | 0.0000 | 1075.0000 |
| Lag_3 | 274.3648 | 164.0000 | 305.1143 | 0.0000 | 1075.0000 |
| Rolling_Mean_3 | 276.9069 | 164.0000 | 297.1950 | 0.0000 | 1075.0000 |
| Rolling_Mean_6 | 270.3328 | 162.8333 | 288.4998 | 0.0000 | 1075.0000 |
| Inventory_Utilization | 0.1188 | 0.0664 | 0.1274 | 0.0000 | 0.6434 |
| Lead_Time_Category | 0.8133 | 1.0000 | 0.7248 | 0.0000 | 2.0000 |
| Demand_Growth | 0.1267 | 0.0000 | 0.6210 | -0.9984 | 2.0000 |
| Inventory_Coverage | 10.5730 | 1.2478 | 17.9365 | 0.0000 | 52.0000 |
| Budget_Utilization | 0.2569 | 0.1309 | 0.3318 | 0.0000 | 2.2909 |
| Supplier_Risk_Score | 0.1551 | 0.1066 | 0.1389 | 0.0042 | 0.6499 |
| Seasonal_Demand_Index | 1.0000 | 0.8691 | 0.3262 | 0.5772 | 1.6993 |
| Transportation_Cost_Index | 0.4950 | 0.4500 | 0.2121 | 0.2000 | 1.2748 |

---

## 9. Final Dataset Quality Assessment
The preprocessed dataset is determined to be of **research-quality** and ready for model training:
1. **Mathematical Stability**: Exploding variables in `Demand_Growth` and `Inventory_Coverage` have been stabilized and capped.
2. **Leakage Protection**: All imputation, scaling, and categorical encoding parameters were fit strictly on the training partition and applied to validation/test partitions chronologically.
3. **Completeness**: No missing values, duplicate records, negative inventories, or invalid categorical classes remain.
