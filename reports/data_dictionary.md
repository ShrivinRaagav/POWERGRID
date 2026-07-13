# POWERGRID Supply Chain Optimization - Data Dictionary

This document details all columns in the processed datasets (`train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`) produced by the preprocessing pipeline.

---

| Column Name | Data Type | Description | Allowed Range / Format | Example Value | Used In Future Module |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Project_ID** | `str` | Unique identifier code for the transmission line project. | PG-PROJ-001 to 025, or PG-PROJ-EMER-001 to 005 (Emergency) | PG-PROJ-005 | Aggregations and budget utilization constraints. |
| **Date** | `datetime64[us]` | Weekly frequency timestamp of the record. | YYYY-MM-DD | 2023-01-01 00:00:00 | Temporal alignment for DWT, EMD, and LSTMs. |
| **Region** | `str` | POWERGRID geographic region code. | NR, ER, WR, SR, NER | ER | Transportation node optimization routes. |
| **State** | `str` | State location where the project and warehouse reside. | e.g. Haryana, West Bengal, Assam, Maharashtra, etc. | West Bengal | Geographical grouping variables. |
| **Warehouse** | `str` | POWERGRID storage warehouse depot. | WH-[REGION]-01, WH-[REGION]-02 | WH-ER-01 | Warehouse capacity balance constraints in optimization. |
| **Supplier** | `str` | Material parts manufacturer/supplier name. | Tata Power, KEC International, Kalpataru Power, Skipper, Sterlite | Sterlite Power | Supplier production capacity allocation constraints. |
| **Material_Type** | `str` | The specific grid equipment or material category. | Conductor, Tower Member, Insulator, Transformer, Earthwire, Hardware Fittings | Conductor | Multi-product demand forecast target index. |
| **Project_Phase** | `str` | Active grid development stage. | Planning, Foundation, Tower Erection, Stringing, Testing & Commissioning | Planning | Predictive indicator for construction material demand shifts. |
| **Tower_Type** | `str` | Model design of transmission line tower being erected. | Suspension, Tension, Terminal, Transposition, Angle | Transposition | Material specification planning. |
| **Substation_Type** | `str` | Type of substation linked to transmission project. | AIS (Air Insulated), GIS (Gas Insulated), Hybrid | GIS (Gas Insulated) | Substation component forecasting. |
| **Historical_Demand** | `float64` | Quantity of the material consumed in the previous period. | >= 0 (units/meters) | 69.0000 | Baseline autoregressive regression parameter. |
| **Current_Inventory** | `float64` | Quantity of the material stored in the warehouse. | 0 to Storage_Capacity | 1969.0000 | Inventory state variables in linear programming optimization. |
| **Lead_Time_Days** | `float64` | Time taken by the supplier to deliver order in days. | 5 to 150 days | 49.0000 | Lead time buffer calculations in inventory replenishment. |
| **Supplier_Risk** | `float64` | Risk rating index representing supplier reliability. | 0.05 to 0.95 (lower is better) | 0.1967 | Constraint penalty weights in multi-objective LP optimization. |
| **Commodity_Price** | `float64` | Market price index for metals (Steel, Aluminum). | 70.0 to 200.0 | 104.7451 | Purchase cost minimization objective function. |
| **Transportation_Cost** | `float64` | Unit transport rate index, affected by season/weather/disruptions. | >= 15.0 | 61.5377 | Transit cost minimization objective function. |
| **Storage_Capacity** | `int64` | Maximum quantity of materials the warehouse can store. | 3000 to 8000 units | 3651 | Storage capacity constraints: Inventory <= Storage_Capacity. |
| **Production_Capacity** | `int64` | Maximum quantity the supplier can manufacture per period. | 2000 to 7000 units | 5904 | Supplier capacity constraints: Orders <= Production_Capacity. |
| **Project_Budget** | `int64` | Overall project line budget allocation. | 100k to 600k (Thousands INR) | 364951 | Period budget spending constraints. |
| **Weather** | `str` | Local weather during the planning period. | Normal, Rainy, Heavy Wind, Extreme Cold, Extreme Heat | Normal | Logistics and shipping delay probability predictors. |
| **Season** | `str` | Meteorological season. | Winter, Summer, Monsoon, Post-Monsoon | Winter | Seasonality indicators for DWT/EMD decomposition. |
| **Quantity_Required** | `int64` | Quantity of material demanded in the period (Target Label). | >= 0 (units/meters) | 74 | Target variable predicted by MLP/LSTM/XGBoost. |
| **Year** | `int32` | Year extracted from Date. | e.g. 2023, 2024 | 2023 | Time regression component. |
| **Month** | `int32` | Month extracted from Date. | 1 to 12 | 1 | Time regression component. |
| **WeekOfYear** | `int32` | Calendar week number. | 1 to 53 | 52 | Time regression component. |
| **DayOfWeek** | `int32` | Day index of the week. | 0 to 6 | 6 | Weekly schedule alignment. |
| **Is_Quarter_End** | `int32` | Indicator if date lands on fiscal quarter end. | 0 or 1 | 0 | Fiscal budget adjustments. |
| **Month_Sin** | `float64` | Sine transform of Month. | -1.0 to 1.0 | 0.5000 | Cyclical predictor for MLP/LSTM/RF. |
| **Month_Cos** | `float64` | Cosine transform of Month. | -1.0 to 1.0 | 0.8660 | Cyclical predictor for MLP/LSTM/RF. |
| **Week_Sin** | `float64` | Sine transform of WeekOfYear. | -1.0 to 1.0 | 0.0000 | Cyclical predictor for MLP/LSTM/RF. |
| **Week_Cos** | `float64` | Cosine transform of WeekOfYear. | -1.0 to 1.0 | 1.0000 | Cyclical predictor for MLP/LSTM/RF. |
| **Lag_1** | `float64` | Demand of material from 1 week ago. | >= 0 | 164.0000 | Autoregressive forecasting inputs. |
| **Lag_2** | `float64` | Demand of material from 2 weeks ago. | >= 0 | 164.0000 | Autoregressive forecasting inputs. |
| **Lag_3** | `float64` | Demand of material from 3 weeks ago. | >= 0 | 164.0000 | Autoregressive forecasting inputs. |
| **Rolling_Mean_3** | `float64` | Rolling average demand of last 3 weeks. | >= 0 | 164.0000 | Moving average forecasting smoothing. |
| **Rolling_Mean_6** | `float64` | Rolling average demand of last 6 weeks. | >= 0 | 164.0000 | Moving average forecasting smoothing. |
| **Inventory_Utilization** | `float64` | Warehouse capacity utilization rate. | 0.0 to 1.0 | 0.5393 | Logistics bottlenecks and storage allocation planning. |
| **Lead_Time_Category** | `int32` | Binned lead time intervals. | 0 (Short), 1 (Medium), 2 (Long) | 1 | Decision parameters for inventory replenishment trigger. |
| **Demand_Growth** | `float64` | Relative change rate between Lag_1 and Lag_2. | -inf to inf | 0.0000 | Trend indicator for gradient boosting regressors. |
| **Inventory_Coverage** | `float64` | Number of periods current stock will cover demand. | 0.0 to inf | 12.0061 | Safety stock optimization constraints. |
| **Budget_Utilization** | `float64` | Estimated spending relative to project budget. | 0.0 to inf | 0.0747 | Financial constraint checks. |
| **Supplier_Risk_Score** | `float64` | Composite supplier risk index. | 0.0 to inf (lower is safer) | 0.0964 | Risk minimization objective function in LP model. |
| **Seasonal_Demand_Index** | `float64` | Historical demand ratio of material in the season. | 0.0 to 2.5 | 1.2272 | Seasonality multipliers for regression models. |
| **Transportation_Cost_Index** | `float64` | Ratio of unit transport cost to commodity price. | 0.0 to inf | 0.5875 | Cost minimization weights. |
