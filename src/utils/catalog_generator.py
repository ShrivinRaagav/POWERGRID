import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from src.utils.helpers import setup_logger

logger = setup_logger("catalog_generator")

# Static dictionary of known baseline columns
FEATURE_METADATA_REGISTRY: Dict[str, Dict[str, str]] = {
    # Raw/Basic fields
    "Project_ID": {
        "Description": "Unique identifier for the transmission line project.",
        "Formula": "Categorical Identifier",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Date": {
        "Description": "Weekly timestamp of the record.",
        "Formula": "ISO-8601 Date",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Region": {
        "Description": "Geographical region code (e.g. NR, ER, WR, SR, NER).",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "State": {
        "Description": "State where the project is located.",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Warehouse": {
        "Description": "Target warehouse identifier.",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Supplier": {
        "Description": "Vendor providing the material.",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Material_Type": {
        "Description": "Material category (e.g. Conductor, Insulator, Tower Member).",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Project_Phase": {
        "Description": "Current phase of project construction.",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Tower_Type": {
        "Description": "Structural type of transmission tower used.",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Substation_Type": {
        "Description": "Substation insulation method (e.g. AIS, GIS).",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Historical_Demand": {
        "Description": "Last month's historical demand baseline.",
        "Formula": "Numeric Value",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Current_Inventory": {
        "Description": "Currently available stock level at the warehouse.",
        "Formula": "Stock count",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Lead_Time_Days": {
        "Description": "Supply duration in days from order to delivery.",
        "Formula": "Duration",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Supplier_Risk": {
        "Description": "Assessed supplier operational risk score (0.0 to 1.0).",
        "Formula": "Probability/Index",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Commodity_Price": {
        "Description": "Market raw material price index.",
        "Formula": "Market Index",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Transportation_Cost": {
        "Description": "Freight shipping tariff index.",
        "Formula": "Cost Index",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Storage_Capacity": {
        "Description": "Warehouse storage limit.",
        "Formula": "Capacity limit",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Production_Capacity": {
        "Description": "Supplier manufacturing capacity limit.",
        "Formula": "Capacity limit",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Project_Budget": {
        "Description": "Remaining financial budget allocation for the project.",
        "Formula": "Financial Amount",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Weather": {
        "Description": "Severe local weather conditions (Extreme Cold, Heat, Normal).",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Season": {
        "Description": "Climatic season (Summer, Monsoon, Winter, etc.).",
        "Formula": "Categorical",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    "Quantity_Required": {
        "Description": "Target variable: actual material demand requested.",
        "Formula": "Demand quantity",
        "Origin": "Raw Data",
        "Module": "Module 1"
    },
    # Temporal & Cyclical
    "Year": {
        "Description": "Calendar year extracted from Date.",
        "Formula": "dt.year",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    "Month": {
        "Description": "Calendar month index (1 to 12) from Date.",
        "Formula": "dt.month",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    "WeekOfYear": {
        "Description": "Calendar week index (1 to 53) from Date.",
        "Formula": "dt.isocalendar().week",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    "DayOfWeek": {
        "Description": "Day index of the week (0 to 6) from Date.",
        "Formula": "dt.dayofweek",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    "Is_Quarter_End": {
        "Description": "Flag indicating if the date lies at the end of a fiscal quarter.",
        "Formula": "dt.is_quarter_end",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    "Month_Sin": {
        "Description": "Sine transformation of month index for cyclical mapping.",
        "Formula": "sin(2 * pi * Month / 12)",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    "Month_Cos": {
        "Description": "Cosine transformation of month index for cyclical mapping.",
        "Formula": "cos(2 * pi * Month / 12)",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    "Week_Sin": {
        "Description": "Sine transformation of week index for cyclical mapping.",
        "Formula": "sin(2 * pi * Week / 52.177)",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    "Week_Cos": {
        "Description": "Cosine transformation of week index for cyclical mapping.",
        "Formula": "cos(2 * pi * Week / 52.177)",
        "Origin": "Date Column",
        "Module": "Module 1"
    },
    # Engineered features
    "Lag_1": {
        "Description": "One-period historical lag of target Quantity_Required.",
        "Formula": "Shift(1) grouped by Warehouse & Material",
        "Origin": "Quantity_Required",
        "Module": "Module 1"
    },
    "Lag_2": {
        "Description": "Two-period historical lag of target Quantity_Required.",
        "Formula": "Shift(2) grouped by Warehouse & Material",
        "Origin": "Quantity_Required",
        "Module": "Module 1"
    },
    "Lag_3": {
        "Description": "Three-period historical lag of target Quantity_Required.",
        "Formula": "Shift(3) grouped by Warehouse & Material",
        "Origin": "Quantity_Required",
        "Module": "Module 1"
    },
    "Rolling_Mean_3": {
        "Description": "3-week rolling average demand shift-1 lag sequence.",
        "Formula": "RollingMean(3) of Lag_1",
        "Origin": "Quantity_Required",
        "Module": "Module 1"
    },
    "Rolling_Mean_6": {
        "Description": "6-week rolling average demand shift-1 lag sequence.",
        "Formula": "RollingMean(6) of Lag_1",
        "Origin": "Quantity_Required",
        "Module": "Module 1"
    },
    "Inventory_Utilization": {
        "Description": "Inventory fill-rate ratio of warehouse capacity.",
        "Formula": "Current_Inventory / Storage_Capacity",
        "Origin": "Inventory, Capacity",
        "Module": "Module 1"
    },
    "Lead_Time_Category": {
        "Description": "Categorized lead time duration index (0: Short, 1: Medium, 2: Long).",
        "Formula": "Cut(Lead_Time_Days, Bins)",
        "Origin": "Lead_Time_Days",
        "Module": "Module 1"
    },
    "Demand_Growth": {
        "Description": "Demand growth speed relative to rolling average.",
        "Formula": "(Lag_1 - Rolling_Mean_3) / (Rolling_Mean_3 + epsilon)",
        "Origin": "Quantity_Required",
        "Module": "Module 1"
    },
    "Inventory_Coverage": {
        "Description": "Warehouse stock replenishment safety margin in weeks.",
        "Formula": "Current_Inventory / (Rolling_Mean_3 + epsilon)",
        "Origin": "Inventory, Lags",
        "Module": "Module 1"
    },
    "Budget_Utilization": {
        "Description": "Remaining financial budget allocation metric.",
        "Formula": "Project_Budget / Capacity",
        "Origin": "Project_Budget",
        "Module": "Module 1"
    },
    "Supplier_Risk_Score": {
        "Description": "Scaled and penalized supplier reliability index.",
        "Formula": "Supplier_Risk * Lead_Time_Days",
        "Origin": "Supplier_Risk, Lead_Time",
        "Module": "Module 1"
    },
    "Seasonal_Demand_Index": {
        "Description": "Mean demand index of material in season relative to overall average.",
        "Formula": "Mean(Material_Season) / Mean(Material_Overall)",
        "Origin": "Material_Type, Season, Target",
        "Module": "Module 1"
    },
    "Transportation_Cost_Index": {
        "Description": "Transportation index scaled against supplier delay risks.",
        "Formula": "Transportation_Cost * Supplier_Risk",
        "Origin": "Transportation_Cost, Risk",
        "Module": "Module 1"
    },
    # Classical signal components
    "Classical_Trend": {
        "Description": "Low-frequency trend component extracted from demand signal.",
        "Formula": "Moving Average filter (statsmodels)",
        "Origin": "Quantity_Required",
        "Module": "Module 2"
    },
    "Classical_Seasonal": {
        "Description": "Additive periodic seasonal component extracted from demand signal.",
        "Formula": "Average of seasonal detrended signals",
        "Origin": "Quantity_Required",
        "Module": "Module 2"
    },
    "Classical_Residual": {
        "Description": "High-frequency irregular remainder component from demand signal.",
        "Formula": "Original - Trend - Seasonal",
        "Origin": "Quantity_Required",
        "Module": "Module 2"
    },
    # EMD Residual
    "EMD_Residual": {
        "Description": "The final monotonic residual trend extracted via EMD sifting.",
        "Formula": "Original - Sum(IMFs)",
        "Origin": "Quantity_Required",
        "Module": "Module 2"
    }
}

def get_dynamic_metadata(feature_name: str) -> Dict[str, str]:
    """Generates descriptions for dynamic features (e.g. IMF_1, cD_2, statistics)."""
    # EMD IMF check
    if feature_name.startswith("EMD_IMF_"):
        try:
            imf_num = int(feature_name.split("_")[-1])
        except:
            imf_num = "?"
        return {
            "Description": f"Intrinsic Mode Function (IMF) #{imf_num} representing oscillatory mode of demand variation.",
            "Formula": f"EMD Sifting Mode #{imf_num}",
            "Origin": "Quantity_Required",
            "Module": "Module 2"
        }
        
    # DWT Details check
    elif feature_name.startswith("DWT_cD_"):
        parts = feature_name.split("_")
        level_str = parts[2] if len(parts) > 2 else "?"
        metric = parts[-1] if len(parts) > 2 else "values"
        
        desc = f"Detail wavelet coefficients at level {level_str}."
        if metric == "Energy":
            desc = f"Sum of squares (energy) of detail wavelet coefficients at level {level_str}."
        elif metric == "Entropy":
            desc = f"Shannon entropy of detail wavelet coefficients at level {level_str}."
            
        return {
            "Description": desc,
            "Formula": f"Discrete Wavelet Transform Filter (Level {level_str})",
            "Origin": "Quantity_Required",
            "Module": "Module 2"
        }
        
    # DWT cA check
    elif feature_name.startswith("DWT_cA_"):
        metric = feature_name.split("_")[-1]
        desc = f"Approximation wavelet coefficients representing low-frequency grid demand profile."
        if metric == "Energy":
            desc = f"Energy (sum of squares) of approximation coefficients cA."
        elif metric == "Entropy":
            desc = f"Shannon entropy of approximation coefficients cA."
            
        return {
            "Description": desc,
            "Formula": "DWT Low-Pass Filter",
            "Origin": "Quantity_Required",
            "Module": "Module 2"
        }
        
    # Statistical features
    elif feature_name in ["Signal_Length", "Signal_Energy", "Signal_Mean_Energy", "Signal_Entropy", "Dominant_Frequency", "Trend_Strength", "Seasonality_Strength", "Trend_Slope", "Residual_Variance", "Residual_Mean", "Residual_Std", "DWT_Approximation_Energy_Ratio", "DWT_Detail_Energy_Ratio", "Num_IMFs", "Dominant_IMF"]:
        return {
            "Description": f"Statistical signal feature representing {feature_name.lower().replace('_', ' ')} of the demand curve.",
            "Formula": "Signal Analysis Statistic",
            "Origin": "Quantity_Required",
            "Module": "Module 2"
        }
        
    elif feature_name.startswith("EMD_IMF_") and feature_name.endswith(("_Energy", "_Mean", "_Std", "_Var")):
        parts = feature_name.split("_")
        imf_num = parts[2]
        metric = parts[3]
        return {
            "Description": f"Statistical metric ({metric}) calculated on EMD Intrinsic Mode Function #{imf_num}.",
            "Formula": f"np.{metric.lower()}(IMF_{imf_num})",
            "Origin": "Quantity_Required",
            "Module": "Module 2"
        }
        
    elif feature_name.startswith("EMD_Resid_"):
        metric = feature_name.split("_")[-1]
        return {
            "Description": f"Statistical metric ({metric}) calculated on the EMD Residual monotonic trend.",
            "Formula": f"np.{metric.lower()}(EMD_Residual)",
            "Origin": "Quantity_Required",
            "Module": "Module 2"
        }
        
    # Default fallback
    return {
        "Description": "Engineered pipeline input feature.",
        "Formula": "N/A",
        "Origin": "Computed",
        "Module": "Module 1"
    }

def generate_feature_catalog(
    all_features: List[str],
    selected_features: List[str],
    output_path: Path
):
    """
    Generates reports/feature_catalog.md dynamically mapping all features in 
    the dataset to their definitions and whether they are active in forecasting.
    """
    logger.info(f"Generating feature catalog dynamically at {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    selected_set = set(selected_features)
    
    rows = []
    # Deduplicate and sort columns
    sorted_features = sorted(list(set(all_features)))
    
    for feat in sorted_features:
        # Resolve metadata
        meta = FEATURE_METADATA_REGISTRY.get(feat)
        if meta is None:
            meta = get_dynamic_metadata(feat)
            
        used = "✔ Yes" if feat in selected_set else "❌ No (Dropped)"
        
        rows.append(
            f"| **{feat}** | {meta['Description']} | `{meta['Formula']}` | {meta['Origin']} | {meta['Module']} | {used} |"
        )
        
    md_header = """# POWERGRID Dynamic Feature Catalog

This catalog documents every feature column generated during the preprocessing (Module 1) and time-series decomposition (Module 2) stages of the pipeline, noting its origin, formula, and whether it was selected for downstream Machine Learning Forecasting (Module 3).

---

## 📋 Feature Definitions Table

| Feature Name | Description | Formula / Transform | Origin | Module | Used in Forecasting? |
| :--- | :--- | :--- | :---: | :---: | :---: |
"""
    
    md_body = "\n".join(rows)
    md_content = md_header + md_body + "\n"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.info(f"Feature catalog successfully generated and saved to {output_path}")
