# Data Dictionary - POWERGRID Material Demand & Supply Chain Optimization

This dictionary defines each field in the raw and processed datasets of the POWERGRID demand forecasting and supply chain optimization project.

---

## 1. Raw Dataset Columns (Original Scope)

| Column Name | Data Type | Range/Format | Description |
| :--- | :--- | :--- | :--- |
| **Project_ID** | Categorical | `PG-PROJ-001` to `030` | Unique identifier for POWERGRID transmission projects. |
| **Date** | DateTime | `YYYY-MM-DD` | Date of the record (weekly frequency). |
| **Region** | Categorical | `NR`, `ER`, `WR`, `SR`, `NER` | Geographical POWERGRID region. |
| **State** | Categorical | e.g. `Haryana`, `Assam`, etc. | State containing the project site and warehouse. |
| **Warehouse** | Categorical | `WH-[REGION]-01` / `02` | Storage depot warehouse. |
| **Supplier** | Categorical | e.g. `Tata Power`, `Sterlite` | Manufacturer/Supplier of the material. |
| **Material_Type** | Categorical | `Conductor`, `Tower Member`, `Insulator`, `Transformer`, `Earthwire`, `Hardware Fittings` | Type of power grid equipment/material. |
| **Project_Phase** | Categorical | `Planning`, `Foundation`, `Tower Erection`, `Stringing`, `Testing & Commissioning` | Active construction phase of the project. |
| **Tower_Type** | Categorical | `Suspension`, `Tension`, `Angle`, etc. | Model of transmission towers used. |
| **Substation_Type** | Categorical | `AIS`, `GIS`, `Hybrid` | Type of substations linked. |
| **Historical_Demand** | Numerical (Int) | `[0, 1000]` | Quantity of material used in previous period. |
| **Current_Inventory** | Numerical (Int) | `[0, Storage_Capacity]` | Current warehouse inventory of the material. |
| **Lead_Time_Days** | Numerical (Int) | `[10, 120]` | Supplier delivery lead time in days. |
| **Supplier_Risk** | Numerical (Float) | `[0.0, 1.0]` | Risk reliability score of the supplier (lower is safer). |
| **Commodity_Price** | Numerical (Float) | `[80.0, 150.0]` | Market commodity price index for raw materials (Steel/Alum). |
| **Transportation_Cost**| Numerical (Float) | `[15.0, 100.0]` | Cost index for transport, influenced by weather/distance. |
| **Storage_Capacity** | Numerical (Int) | `[2000, 10000]` | Max inventory limit of the warehouse. |
| **Production_Capacity**| Numerical (Int) | `[1500, 8000]` | Max production quantity the supplier can produce per period. |
| **Project_Budget** | Numerical (Float) | `[100k, 500k]` | Allocated transmission line budget (Thousands INR). |
| **Weather** | Categorical | `Normal`, `Rainy`, `Heavy Wind`, `Extreme Cold`, `Extreme Heat` | Local weather during the planning period. |
| **Season** | Categorical | `Winter`, `Summer`, `Monsoon`, `Post-Monsoon` | Meteorological season. |
| **Quantity_Required** | Numerical (Int) | `[0, 1500]` | **Target Label**: Quantity of material demanded in the period. |

---

## 2. Engineered Temporal & Cyclical Features

These features extract calendar dependencies. Sin/Cos values capture cyclical patterns (e.g. Month 12 is close to Month 1).

| Column Name | Data Type | Range/Format | Description |
| :--- | :--- | :--- | :--- |
| **Year** | Numerical (Int) | `2023`, `2024`, `2025` | Extract of year from planning date. |
| **Month** | Numerical (Int) | `[1, 12]` | Extract of month from planning date. |
| **WeekOfYear** | Numerical (Int) | `[1, 53]` | Extract of calendar week number. |
| **DayOfWeek** | Numerical (Int) | `[0, 6]` | Day of the week (Monday=0, Sunday=6). |
| **Is_Quarter_End** | Binary (Int) | `0` or `1` | Indicator if date is at the end of a fiscal quarter. |
| **Month_Sin** | Numerical (Float) | `[-1.0, 1.0]` | Sine component of month cyclical index. |
| **Month_Cos** | Numerical (Float) | `[-1.0, 1.0]` | Cosine component of month cyclical index. |
| **Week_Sin** | Numerical (Float) | `[-1.0, 1.0]` | Sine component of week cyclical index. |
| **Week_Cos** | Numerical (Float) | `[-1.0, 1.0]` | Cosine component of week cyclical index. |

---

## 3. Engineered Time-Series & Domain Features

Features created to support future predictive models (Random Forest, XGBoost, LSTM) and optimize supply chain paths.

| Column Name | Data Type | Range/Format | Description |
| :--- | :--- | :--- | :--- |
| **Lag_1** | Numerical (Float) | `[0, 1500]` | Demand (`Quantity_Required`) from previous period. |
| **Lag_2** | Numerical (Float) | `[0, 1500]` | Demand from 2 periods ago. |
| **Lag_3** | Numerical (Float) | `[0, 1500]` | Demand from 3 periods ago. |
| **Rolling_Mean_3** | Numerical (Float) | `[0, 1500]` | Rolling average demand of last 3 periods (excluding current). |
| **Rolling_Mean_6** | Numerical (Float) | `[0, 1500]` | Rolling average demand of last 6 periods (excluding current). |
| **Inventory_Utilization**| Numerical (Float) | `[0.0, 1.0]` | Ratio of warehouse space utilized: `Current_Inventory / Storage_Capacity`. |
| **Lead_Time_Category** | Categorical (Int)| `0`, `1`, `2` | Binned lead times: Short (<35d)=0, Medium (35-70d)=1, Long (>70d)=2. |
| **Demand_Growth** | Numerical (Float) | `[-inf, inf]` | Relative growth of demand: `(Lag_1 - Lag_2) / (Lag_2 + epsilon)`. |
| **Inventory_Coverage** | Numerical (Float) | `[0.0, inf]` | Periods current inventory lasts: `Current_Inventory / Rolling_Mean_3`. |
| **Budget_Utilization** | Numerical (Float) | `[0.0, inf]` | Value of lagged demand relative to project budget. |
| **Supplier_Risk_Score**| Numerical (Float) | `[0.0, 1.0]` | Composite metric multiplying risk rating and lead time proportion. |
| **Seasonal_Demand_Index**| Numerical (Float)| `[0.0, 2.5]` | Historical demand multiplier for the specific material in the season. |
| **Transportation_Cost_Index**| Numerical (Float)| `[0.0, inf]` | Shipping cost ratio: `Transportation_Cost / Commodity_Price`. |
