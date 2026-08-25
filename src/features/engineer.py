import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
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
        self.material_median_demand: Dict[str, float] = {}
        self.is_fit = False

    def fit(self, train_df: pd.DataFrame):
        """
        Fits feature engineering parameters on the training dataset.
        For example, calculates the Seasonal Demand Index mapping and material-level medians.
        """
        logger.info("Fitting FeatureEngineer on training dataset...")
        df = train_df.copy()
        
        # Ensure Date is datetime
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        
        # Calculate global & material-specific median demand for robust imputation
        self.global_median_demand = float(df[TARGET_COL].median()) if TARGET_COL in df.columns else 0.0
        if "Material_Type" in df.columns and TARGET_COL in df.columns:
            self.material_median_demand = df.groupby("Material_Type")[TARGET_COL].median().to_dict()
        else:
            self.material_median_demand = {}
        
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

    def transform(self, df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Engineers features on the dataframe.
        If history_df is provided (e.g. preceding chronological split data),
        it is prepended to maintain continuous time-series lag context without boundary distortion.
        """
        if not self.is_fit:
            raise ValueError("FeatureEngineer must be fit on training data before transforming.")
            
        logger.info(f"Engineering features on dataset of shape {df.shape}...")
        
        # Prepend preceding history if available to maintain continuous lag context
        if history_df is not None and not history_df.empty:
            df_combined = pd.concat([history_df, df], ignore_index=True)
            split_offset = len(history_df)
        else:
            df_combined = df.copy()
            split_offset = 0
            
        # Ensure correct date sorting for lag/rolling calculations
        df_combined[DATE_COL] = pd.to_datetime(df_combined[DATE_COL])
        df_combined["_orig_seq_idx"] = np.arange(len(df_combined))
        df_combined = df_combined.sort_values(by=["Warehouse", "Material_Type", DATE_COL]).reset_index(drop=True)
        
        # --- 1. Lags & Rolling features (Grouped by Warehouse and Material_Type) ---
        group_cols = ["Warehouse", "Material_Type"]
        
        # Lag demand (use shift(1) to avoid lookahead bias of the current period's target demand)
        df_combined["Lag_1"] = df_combined.groupby(group_cols)[TARGET_COL].shift(1)
        df_combined["Lag_2"] = df_combined.groupby(group_cols)[TARGET_COL].shift(2)
        df_combined["Lag_3"] = df_combined.groupby(group_cols)[TARGET_COL].shift(3)
        
        # Rolling averages & extremes (on shift(1) to avoid leakage)
        df_combined["Rolling_Mean_3"] = df_combined.groupby(group_cols)["Lag_1"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df_combined["Rolling_Mean_6"] = df_combined.groupby(group_cols)["Lag_1"].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
        df_combined["Rolling_Max_3"] = df_combined.groupby(group_cols)["Lag_1"].transform(
            lambda x: x.rolling(window=3, min_periods=1).max()
        )
        df_combined["Rolling_Min_3"] = df_combined.groupby(group_cols)["Lag_1"].transform(
            lambda x: x.rolling(window=3, min_periods=1).min()
        )
        df_combined["Rolling_STD_3"] = df_combined.groupby(group_cols)["Lag_1"].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        ).fillna(0.0)
        
        # Impute NaNs created by shifting/rolling with material-specific median (fallback to global median)
        mat_fill_series = df_combined["Material_Type"].map(self.material_median_demand).fillna(self.global_median_demand)
        for col in ["Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3", "Rolling_Mean_6", "Rolling_Max_3", "Rolling_Min_3", "Rolling_STD_3"]:
            df_combined[col] = df_combined[col].fillna(mat_fill_series)
            
        # --- 2. High-Frequency Momentum, Velocity & Volatility Features ---
        # Rate of demand change (velocity)
        df_combined["Demand_Velocity_1"] = df_combined["Lag_1"] - df_combined["Lag_2"]
        df_combined["Demand_Velocity_2"] = df_combined["Lag_2"] - df_combined["Lag_3"]
        # Second derivative (acceleration / momentum shift)
        df_combined["Demand_Acceleration"] = df_combined["Demand_Velocity_1"] - df_combined["Demand_Velocity_2"]
        # Rolling coefficient of variation (volatility ratio)
        df_combined["Demand_Volatility_3"] = df_combined["Rolling_STD_3"] / (df_combined["Rolling_Mean_3"] + 1e-5)
        # Leading Peak/Dip indicator triggers
        df_combined["Peak_Spike_Flag"] = (df_combined["Lag_1"] > (df_combined["Rolling_Mean_3"] * 1.25)).astype(np.float32)
        df_combined["Dip_Drop_Flag"] = (df_combined["Lag_1"] < (df_combined["Rolling_Mean_3"] * 0.75)).astype(np.float32)

        # --- 3. Domain Engineered Features ---
        # Inventory Utilization
        df_combined["Inventory_Utilization"] = df_combined["Current_Inventory"] / (df_combined["Storage_Capacity"] + 1e-5)
        
        # Lead Time Category (Short: 0, Medium: 1, Long: 2)
        df_combined["Lead_Time_Category"] = pd.cut(
            df_combined["Lead_Time_Days"],
            bins=[-np.inf, 35, 70, np.inf],
            labels=[0, 1, 2]
        ).astype(np.int32)
        
        # Demand Growth Rate: stabilized and clipped
        df_combined["Demand_Growth"] = np.clip((df_combined["Lag_1"] - df_combined["Lag_2"]) / (df_combined["Lag_2"] + 1.0), -2.0, 2.0)
        
        # Inventory Coverage: Current Inventory / (Rolling_Mean_3 + 1.0)
        df_combined["Inventory_Coverage"] = np.clip(df_combined["Current_Inventory"] / (df_combined["Rolling_Mean_3"] + 1.0), 0.0, 52.0)
        
        # Budget Utilization: (Lag_1 * (Price + Transport)) / Project Budget
        est_cost_per_unit = df_combined["Commodity_Price"] + df_combined["Transportation_Cost"]
        df_combined["Budget_Utilization"] = (df_combined["Lag_1"] * est_cost_per_unit) / (df_combined["Project_Budget"] + 1e-5)
        
        # Supplier Risk Score: Combine risk probability and lead time
        df_combined["Supplier_Risk_Score"] = df_combined["Supplier_Risk"] * (df_combined["Lead_Time_Days"] / 100.0)
        
        # Seasonal Demand Index
        seasonal_indices = []
        for _, row in df_combined.iterrows():
            mat = row.get("Material_Type", "")
            seas = row.get("Season", "")
            index_val = self.seasonal_demand_index_map.get((mat, seas), 1.0)
            seasonal_indices.append(index_val)
        df_combined["Seasonal_Demand_Index"] = seasonal_indices
        
        # Transportation Cost Index: Ratio of transport cost to commodity price
        df_combined["Transportation_Cost_Index"] = df_combined["Transportation_Cost"] / (df_combined["Commodity_Price"] + 1e-5)
        
        # Restore sequence and slice off history if prepended
        df_combined = df_combined.sort_values(by="_orig_seq_idx").drop(columns=["_orig_seq_idx"]).reset_index(drop=True)
        if split_offset > 0:
            df_out = df_combined.iloc[split_offset:].reset_index(drop=True)
        else:
            df_out = df_combined
            
        logger.info(f"Feature engineering completed. Output shape: {df_out.shape}")
        return df_out

def generate_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a statistical summary of engineered features.
    """
    engineered_cols = [
        "Lag_1", "Lag_2", "Lag_3", "Rolling_Mean_3", "Rolling_Mean_6",
        "Rolling_Max_3", "Rolling_Min_3", "Rolling_STD_3",
        "Demand_Velocity_1", "Demand_Velocity_2", "Demand_Acceleration",
        "Demand_Volatility_3", "Peak_Spike_Flag", "Dip_Drop_Flag",
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
