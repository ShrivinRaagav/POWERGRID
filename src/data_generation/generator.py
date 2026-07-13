import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from src.config.settings import (
    RAW_DATA_PATH, RANDOM_SEED, VALID_REGIONS, VALID_PROJECT_PHASES, VALID_MATERIAL_TYPES
)
from src.utils.helpers import setup_logger

logger = setup_logger("generator")

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_powergrid_data(num_rows: int = 6000) -> pd.DataFrame:
    """
    Generates a realistic synthetic POWERGRID dataset for transmission line material planning.
    Includes controlled anomalies (missing values, negative values, duplicates, invalid values)
    to test the cleaning and validation pipelines.
    """
    logger.info(f"Starting synthetic dataset generation for {num_rows} rows...")
    
    # Establish geographical mappings
    region_state_wh = {
        "NR": [("Haryana", "WH-NR-01"), ("Uttar Pradesh", "WH-NR-02")],
        "ER": [("West Bengal", "WH-ER-01"), ("Bihar", "WH-ER-02")],
        "WR": [("Maharashtra", "WH-WR-01"), ("Gujarat", "WH-WR-02")],
        "SR": [("Tamil Nadu", "WH-SR-01"), ("Andhra Pradesh", "WH-SR-02")],
        "NER": [("Assam", "WH-NER-01"), ("Manipur", "WH-NER-02")]
    }
    
    # Supplier specialization
    suppliers = {
        "Tata Power": ["Transformer", "Hardware Fittings"],
        "KEC International": ["Tower Member", "Conductor"],
        "Kalpataru Power": ["Tower Member", "Earthwire"],
        "Skipper Limited": ["Insulator", "Hardware Fittings"],
        "Sterlite Power": ["Conductor", "Earthwire"]
    }
    supplier_list = list(suppliers.keys())
    
    # Tower and Substation types
    tower_types = ["Suspension", "Tension", "Terminal", "Transposition", "Angle"]
    substation_types = ["AIS (Air Insulated)", "GIS (Gas Insulated)", "Hybrid"]
    
    # Setup timeline: 156 weeks (3 years)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(156)]
    
    # Projects
    projects = [f"PG-PROJ-{i:03d}" for i in range(1, 31)]
    project_metadata = {}
    for proj in projects:
        reg = random.choice(list(region_state_wh.keys()))
        state, wh = random.choice(region_state_wh[reg])
        project_metadata[proj] = {
            "Region": reg,
            "State": state,
            "Warehouse": wh,
            "Budget": random.randint(100000, 500000)  # in Thousands INR
        }
        
    # Generate Commodity Price series (Steel/Aluminum index) - Random walk
    commodity_prices = []
    current_price = 100.0
    for _ in range(len(dates)):
        current_price += np.random.normal(0.2, 1.5)  # slight upward drift
        commodity_prices.append(max(80.0, current_price))
        
    data_rows = []
    
    # Warehouse & Material combinations
    wh_materials = []
    for reg, locations in region_state_wh.items():
        for state, wh in locations:
            for mat in VALID_MATERIAL_TYPES:
                wh_materials.append((reg, state, wh, mat))
                
    # We loop over combinations and dates to build sequential records
    total_combinations = len(wh_materials)
    rows_per_comb = num_rows // total_combinations + 1
    
    logger.info(f"Generating time series for {total_combinations} Warehouse-Material groups over {len(dates)} dates...")
    
    row_count = 0
    for reg, state, wh, mat in wh_materials:
        # Determine supplier based on material specialization
        matching_suppliers = [sup for sup, mats in suppliers.items() if mat in mats]
        supplier = random.choice(matching_suppliers) if matching_suppliers else random.choice(supplier_list)
        
        # Base capacities
        storage_capacity = random.randint(2000, 10000)
        production_capacity = random.randint(1500, 8000)
        
        # Lead time details
        base_lead_time = {
            "Transformer": 90,
            "Conductor": 45,
            "Tower Member": 60,
            "Insulator": 30,
            "Earthwire": 30,
            "Hardware Fittings": 20
        }.get(mat, 30)
        
        # Select active projects for this warehouse
        wh_projects = [p for p, meta in project_metadata.items() if meta["Warehouse"] == wh]
        if not wh_projects:
            wh_projects = [random.choice(projects)]
            
        current_inventory = random.randint(500, 3000)
        
        for date_idx, dt in enumerate(dates):
            if row_count >= num_rows:
                break
                
            proj = random.choice(wh_projects)
            proj_budget = project_metadata[proj]["Budget"]
            
            # Determine Season
            month = dt.month
            if month in [12, 1, 2]:
                season = "Winter"
            elif month in [3, 4, 5]:
                season = "Summer"
            elif month in [6, 7, 8, 9]:
                season = "Monsoon"
            else:
                season = "Post-Monsoon"
                
            # Determine Weather based on region and season
            weather_probs = {
                "Winter": {"NR": [0.6, 0.05, 0.05, 0.3, 0.0], "ER": [0.8, 0.05, 0.05, 0.1, 0.0]},
                "Monsoon": {"NER": [0.1, 0.7, 0.15, 0.0, 0.05], "SR": [0.3, 0.5, 0.15, 0.0, 0.05]},
                "Summer": {"WR": [0.5, 0.0, 0.1, 0.0, 0.4], "NR": [0.5, 0.0, 0.1, 0.0, 0.4]}
            }
            # Default weather probability [Normal, Rainy, Heavy Wind, Extreme Cold, Extreme Heat]
            p_dist = [0.7, 0.1, 0.1, 0.05, 0.05]
            if season in weather_probs and reg in weather_probs[season]:
                p_dist = weather_probs[season][reg]
            weather = np.random.choice(["Normal", "Rainy", "Heavy Wind", "Extreme Cold", "Extreme Heat"], p=p_dist)
            
            # Project Phase sequence over 3 years
            phase_idx = int((date_idx / len(dates)) * 5)
            phase_idx = min(4, max(0, phase_idx))
            project_phase = VALID_PROJECT_PHASES[phase_idx]
            
            # Material Type & Phase interaction for demand
            # Demand is high during specific construction phases
            phase_multiplier = 1.0
            if project_phase == "Foundation" and mat in ["Tower Member", "Hardware Fittings"]:
                phase_multiplier = 1.5
            elif project_phase == "Tower Erection" and mat == "Tower Member":
                phase_multiplier = 2.5
            elif project_phase == "Stringing" and mat in ["Conductor", "Insulator", "Earthwire", "Hardware Fittings"]:
                phase_multiplier = 3.0
            elif project_phase == "Testing & Commissioning" and mat == "Transformer":
                phase_multiplier = 2.0
            elif project_phase == "Planning":
                phase_multiplier = 0.2
                
            # Seasonality multiplier (monsoon slows down transmission construction)
            seasonal_multiplier = 1.0
            if season == "Monsoon":
                seasonal_multiplier = 0.4
            elif season == "Winter" and reg == "NR": # Extreme cold slows down north
                seasonal_multiplier = 0.7
                
            # Base demand level
            base_demand = {
                "Conductor": 500,
                "Tower Member": 400,
                "Insulator": 300,
                "Transformer": 5,
                "Earthwire": 150,
                "Hardware Fittings": 200
            }.get(mat, 100)
            
            # Calculate Quantity Required with trend, seasonality, project phase, and random noise
            trend = 1.0 + (date_idx / len(dates)) * 0.3  # 30% demand growth over 3 years
            noise = np.random.normal(0, 0.1 * base_demand)
            quantity_required = int(base_demand * phase_multiplier * seasonal_multiplier * trend + noise)
            quantity_required = max(0, quantity_required)
            
            # Historical Demand (lagged demand with noise)
            hist_noise = np.random.normal(0, 0.05 * base_demand)
            historical_demand = max(0, int(quantity_required * 0.95 + hist_noise))
            
            # Simulating current inventory
            # Inventory decreases with demand, increases with supply
            if date_idx > 0:
                # simple stock replenishment logic
                replenishment = 0
                if current_inventory < quantity_required:
                    replenishment = int(quantity_required * 1.5)
                current_inventory = current_inventory - quantity_required + replenishment
                current_inventory = max(0, min(storage_capacity, current_inventory))
                
            lead_time = base_lead_time + random.randint(-5, 10)
            supplier_risk = float(np.clip(0.1 + (lead_time / 150) * 0.5 + np.random.normal(0, 0.05), 0.05, 0.95))
            commodity_price = commodity_prices[date_idx]
            
            # Transportation cost (higher distance, worse weather)
            weather_trans_mult = 1.0
            if weather in ["Rainy", "Heavy Wind"]:
                weather_trans_mult = 1.25
            elif weather in ["Extreme Cold", "Extreme Heat"]:
                weather_trans_mult = 1.1
            
            base_trans_cost = random.randint(15, 45)
            trans_cost = float(base_trans_cost * weather_trans_mult * (commodity_price / 100.0))
            
            tower = np.random.choice(tower_types)
            substation = np.random.choice(substation_types)
            
            data_rows.append({
                "Project_ID": proj,
                "Date": dt.strftime("%Y-%m-%d"),
                "Region": reg,
                "State": state,
                "Warehouse": wh,
                "Supplier": supplier,
                "Material_Type": mat,
                "Project_Phase": project_phase,
                "Tower_Type": tower,
                "Substation_Type": substation,
                "Historical_Demand": historical_demand,
                "Current_Inventory": current_inventory,
                "Lead_Time_Days": lead_time,
                "Supplier_Risk": supplier_risk,
                "Commodity_Price": commodity_price,
                "Transportation_Cost": trans_cost,
                "Storage_Capacity": storage_capacity,
                "Production_Capacity": production_capacity,
                "Project_Budget": proj_budget,
                "Weather": weather,
                "Season": season,
                "Quantity_Required": quantity_required
            })
            row_count += 1
            
    df = pd.DataFrame(data_rows)
    
    # Apply controlled noise / dirty data (~2% of records)
    logger.info("Injecting controlled anomalies for cleaning & validation verification...")
    
    # 1. Missing values (NaNs) - ~1% of dataset in random spots
    cols_to_null = ["Lead_Time_Days", "Historical_Demand", "Current_Inventory", "Weather", "Supplier_Risk"]
    for col in cols_to_null:
        null_indices = df.sample(frac=0.008, random_state=42).index
        df.loc[null_indices, col] = np.nan
        
    # 2. Negative values in numerical fields (~0.5%)
    neg_inv_indices = df.sample(frac=0.003, random_state=42).index
    df.loc[neg_inv_indices, "Current_Inventory"] = -100
    
    neg_demand_indices = df.sample(frac=0.003, random_state=43).index
    df.loc[neg_demand_indices, "Quantity_Required"] = -50
    df.loc[neg_demand_indices, "Historical_Demand"] = -50
    
    # 3. Invalid dates (~0.5%)
    invalid_date_indices = df.sample(frac=0.004, random_state=44).index
    df.loc[invalid_date_indices, "Date"] = "202-13-45"  # completely invalid date string
    
    # 4. Invalid categoricals (~0.5%)
    invalid_reg_indices = df.sample(frac=0.003, random_state=45).index
    df.loc[invalid_reg_indices, "Region"] = "XX"  # Invalid region
    
    invalid_phase_indices = df.sample(frac=0.003, random_state=46).index
    df.loc[invalid_phase_indices, "Project_Phase"] = "Unknown Phase"  # Invalid project phase
    
    # 5. Duplicates (~1.0% duplicated records)
    dup_indices = df.sample(frac=0.01, random_state=47).index
    df_dups = df.loc[dup_indices].copy()
    df = pd.concat([df, df_dups], ignore_index=True)
    
    # Shuffling the dataset to mix duplicates and normal records
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    logger.info(f"Synthetic dataset generated. Shape: {df.shape}")
    return df

def main():
    df = generate_powergrid_data(6000)
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Saved raw dataset to {RAW_DATA_PATH}")

if __name__ == "__main__":
    main()
