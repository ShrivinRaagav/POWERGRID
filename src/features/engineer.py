import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.config.settings import DATE_COL, TARGET_COL
from src.utils.helpers import setup_logger

logger = setup_logger("engineer")

class FeatureEngineer:
    """
    Modular feature engineering class that:
    1. Computes time-series lag and rolling average features grouped by Warehouse & Material.
    2. Calculates domain supply chain metrics (Inventory Utilization, Inventory Coverage, Demand Growth).
    3. Calculates index features (Supplier Risk Score, Transportation Cost Index).
    4. Computes a Seasonal Demand Index fit on training set to avoid leakage.
    """
    def __init__(self):
        self.seasonal_demand_index_map: Dict[Tuple[str, str], float] = {}
        self.global_median_demand = 0.0
        self.is_fit = False

    def fit(self, train_df: pd.DataFrame):
        """
        Fits feature engineering parameters on the training dataset.
        For example, calculates the Seasonal Demand Index mapping.
        """
        logger.info("Fitting FeatureEngineer on training dataset...")
        df = train_df.copy()
        
        # Ensure Date is datetime
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        
        # Calculate global median demand for general imputation fallback
        self.global_median_demand = df[TARGET_COL].median() if TARGET_COL in df.columns else 0.0
        
        # Compute Seasonal Demand Index: Average demand of material in a season / Average demand of material overall
        if "Season" in df.columns and "Material_Type" in df.columns and TARGET_COL in df.columns:
            # Group by Material and Season
            mat_season_avg = df.groupby(["Material_Type", "Season"])[TARGET_COL].mean().reset_index()
            # Group by Material overall
            mat_overall_avg = df.groupby("Material_Type")[TARGET_COL].mean().reset_index()
            
            # Rename columns
            mat_season_avg = mat_season_avg.rename(columns={TARGET_COL: "Season_Avg"})
            mat_overall_avg = mat_overall_avg.rename(columns={TARGET_COL: "Overall_Avg"})
            
            # Merge
            index_df = pd.merge(mat_season_avg, mat_overall_avg, on="Material_Type")
            index_df["Seasonal_Index"] = index_df["Season_Avg"] / (index_df["Overall_Avg"] + 1e-5)
            
            # Store in map
            for _, row in index_df.iterrows():
                self.seasonal_demand_index_map[(row["Material_Type"], row["Season"])] = float(row["Seasonal_Index"])
                
        self.is_fit = True
        logger.info("FeatureEngineer fitted successfully.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers features on the dataframe.
        """
        if not self.is_fit:
            raise ValueError("FeatureEngineer must be fit on training data before transforming.")
            
        logger.info(f"Engineering features on dataset of shape {df.shape}...")
        df_out = df.copy()
        
        # Ensure correct date sorting for lag/rolling calculations
        df_out[DATE_COL] = pd.to_datetime(df_out[DATE_COL])
        df_out = df_out.sort_values(by=["Warehouse", "Material_Type", DATE_COL]).reset_index(drop=True)
        
        # --- 1. Lags & Rolling features (Grouped by Warehouse and Material_Type) ---
        group_cols = ["Warehouse", "Material_Type"]
        
        # Lag demand (use shift(1) to avoid lookahead bias of the current period's target demand)
        df_out["Lag_1"] = df_out.groupby(group_cols)[TARGET_COL].shift(1)
        df_out["Lag_2"] = df_out.groupby(group_cols)[TARGET_COL].shift(2)
        df_out["Lag_3"] = df_out.groupby(group_cols)[TARGET_COL].shift(3)
        
        # Rolling averages (on shift(1) to avoid leakage)
        df_out["Rolling_Mean_3"] = df_out.groupby(group_cols)["Lag_1"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_out["Rolling_Mean_6"] = df_out.groupby(group_cols)["Lag_1"].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
        
        # Impute NaNs created by shifting/rolling with the material median or global median
        for col in ["Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3", "Rolling_Mean_6"]:
            # Fill with global median demand if still missing
            df_out[col] = df_out[col].fillna(self.global_median_demand)
            
        # --- 2. Domain Engineered Features ---
        
        # Inventory Utilization
        df_out["Inventory_Utilization"] = df_out["Current_Inventory"] / (df_out["Storage_Capacity"] + 1e-5)
        
        # Lead Time Category (Short: 0, Medium: 1, Long: 2)
        df_out["Lead_Time_Category"] = pd.cut(
            df_out["Lead_Time_Days"],
            bins=[-np.inf, 35, 70, np.inf],
            labels=[0, 1, 2]
        ).astype(np.int32)
        
        # Demand Growth Rate: (Lag_1 - Lag_2) / (Lag_2 + 1.0) stabilized and clipped to prevent exploding values
        df_out["Demand_Growth"] = np.clip((df_out["Lag_1"] - df_out["Lag_2"]) / (df_out["Lag_2"] + 1.0), -2.0, 2.0)
        
        # Inventory Coverage: Current Inventory / (Rolling_Mean_3 + 1.0) stabilized and clipped to maximum 52 weeks (1 year)
        df_out["Inventory_Coverage"] = np.clip(df_out["Current_Inventory"] / (df_out["Rolling_Mean_3"] + 1.0), 0.0, 52.0)
        
        # Budget Utilization: (Lag_1 * (Price + Transport)) / Project Budget
        # Using Lag_1 demand here is realistic for feature engineering to avoid lookahead leakage
        est_cost_per_unit = df_out["Commodity_Price"] + df_out["Transportation_Cost"]
        df_out["Budget_Utilization"] = (df_out["Lag_1"] * est_cost_per_unit) / (df_out["Project_Budget"] + 1e-5)
        
        # Supplier Risk Score: Combine risk probability and lead time
        df_out["Supplier_Risk_Score"] = df_out["Supplier_Risk"] * (df_out["Lead_Time_Days"] / 100.0)
        
        # Seasonal Demand Index
        # Lookup seasonal index from the map, fallback to 1.0 (no seasonal effect) if not found
        seasonal_indices = []
        for _, row in df_out.iterrows():
            mat = row.get("Material_Type", "")
            seas = row.get("Season", "")
            index_val = self.seasonal_demand_index_map.get((mat, seas), 1.0)
            seasonal_indices.append(index_val)
        df_out["Seasonal_Demand_Index"] = seasonal_indices
        
        # Transportation Cost Index: Ratio of transport cost to commodity price
        df_out["Transportation_Cost_Index"] = df_out["Transportation_Cost"] / (df_out["Commodity_Price"] + 1e-5)
        
        logger.info(f"Feature engineering completed. Output shape: {df_out.shape}")
        return df_out

def generate_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a statistical summary of engineered features.
    """
    engineered_cols = [
        "Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3", "Rolling_Mean_6",
        "Inventory_Utilization", "Lead_Time_Category", "Demand_Growth",
        "Inventory_Coverage", "Budget_Utilization", "Supplier_Risk_Score",
        "Seasonal_Demand_Index", "Transportation_Cost_Index"
    ]
    
    summary_rows = []
    for col in engineered_cols:
        if col in df.columns:
            series = df[col]
            summary_rows.append({
                "Feature_Name": col,
                "Mean": float(series.mean()),
                "Median": float(series.median()),
                "Std_Dev": float(series.std()),
                "Min": float(series.min()),
                "Max": float(series.max()),
                "Missing_Count": int(series.isna().sum())
            })
            
    return pd.DataFrame(summary_rows)
