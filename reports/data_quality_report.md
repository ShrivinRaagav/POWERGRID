# POWERGRID Supply Chain Optimization - Data Quality Report

This report analyzes the raw grid logistics material planning dataset and validates the changes applied by the data preparation pipeline.

---

## 1. Executive Summary

| Metrics | Raw Dataset | Preprocessed / Cleaned Dataset | Change / Resolution |
| :--- | :--- | :--- | :--- |
| **Row Count** | 6060 | 4199 | Lost 1861 rows due to invalid dates and duplicates. |
| **Column Count** | 22 | 44 | Increased from 22 to 44 after feature engineering. |
| **Missing Values** | 225 | 0 | 100% missing values imputed using training median/mode. |
| **Duplicate Rows** | 60 | 0 | Staged duplicate records removed. |

---

## 2. Preprocessing & Outlier Summary

### Outlier Handling (Winsorization)
To prevent extreme values from distorting future forecasting models (SVR, MLP, LSTM), numerical variables were capped using Interquartile Range (IQR) thresholds $[Q1 - 1.5\times\text{IQR}, Q3 + 1.5\times\text{IQR}]$ fit on the training partition:

- **Historical_Demand**: 227 values adjusted/capped.
- **Current_Inventory**: 104 values adjusted/capped.
- **Lead_Time_Days**: 87 values adjusted/capped.
- **Supplier_Risk**: 293 values adjusted/capped.
- **Transportation_Cost**: 260 values adjusted/capped.

---

## 3. Preprocessed Feature Statistics

The following statistics describe the engineered features computed on the training partition:

| Feature Name | Mean | Median | Std Dev | Min | Max |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Lag_1 | 281.7226 | 164.0000 | 311.5279 | 0.0000 | 1075.0000 |
| Lag_2 | 278.0069 | 164.0000 | 308.2889 | 0.0000 | 1075.0000 |
| Lag_3 | 274.3648 | 164.0000 | 305.1143 | 0.0000 | 1075.0000 |
| Rolling_Mean_3 | 276.9069 | 164.0000 | 297.1950 | 0.0000 | 1075.0000 |
| Rolling_Mean_6 | 270.3328 | 162.8333 | 288.4998 | 0.0000 | 1075.0000 |
| Inventory_Utilization | 0.1188 | 0.0664 | 0.1274 | 0.0000 | 0.6434 |
| Lead_Time_Category | 0.8133 | 1.0000 | 0.7248 | 0.0000 | 2.0000 |
| Demand_Growth | 146987.9465 | 0.0000 | 2324812.7522 | -1.0000 | 107500000.0000 |
| Inventory_Coverage | 6341646.5873 | 1.2734 | 34849848.7728 | 0.0000 | 243350000.0000 |
| Budget_Utilization | 0.2348 | 0.1210 | 0.3009 | 0.0000 | 2.0541 |
| Supplier_Risk_Score | 0.1551 | 0.1066 | 0.1389 | 0.0042 | 0.6499 |
| Seasonal_Demand_Index | 1.0000 | 0.8691 | 0.3262 | 0.5772 | 1.6993 |
| Transportation_Cost_Index | 0.4950 | 0.4500 | 0.2121 | 0.2000 | 1.2748 |

---

## 4. Pipeline Validation Check Log

Below is the record of validation tests run on the raw data (pre-cleaning) and the cleaned data (post-cleaning):

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

## 5. Research Recommendations for Next Phase

1. **Discrete Wavelet Transform (DWT)**: Lags (`Lag_1`, `Lag_2`, `Lag_3`) and rolling means (`Rolling_Mean_3`, `Rolling_Mean_6`) are grouped at the `[Warehouse, Material_Type]` level. Apply DWT or EMD on each series individually to decompose high-frequency supply noises from long-term construction trends.
2. **Sequential Formatting**: For deep learning models (LSTM), shape the preprocessed sequence splits (`train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`) into overlapping sliding windows (e.g. timesteps=4).
3. **Linear Programming Optimization**: The output values of `Supplier_Risk_Score`, `Transportation_Cost_Index`, and predictions of `Quantity_Required` should be passed to `PuLP` to solve the multi-objective linear programming model via the CBC solver.
