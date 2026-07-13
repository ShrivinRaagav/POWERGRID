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
    Simulates real-world operational scenarios:
    - Monsoon season transport delays and cost spikes (June-Sept, ER/NER)
    - Regional weather impacts (NR extreme cold, WR/NR extreme heat waves)
    - Emergency transmission projects (double demand, halved lead time, doubled transport cost)
    - Material shortages (Conductor/Transformer supply bottlenecks in 2024)
    - Inflation and commodity price cycles
    - Public holidays and festivals (Diwali / Winter slow periods)
    - Project acceleration events
    - Inventory stockout logistics penalties
    - Warehouse congestion overhead (>85% capacity)
    - Transit strikes and landslide disruptions
    
    Also injects controlled anomalies (missing/negative values, invalid classes)
    for validator and cleaner testing.
    """
    logger.info(f"Generating realistic POWERGRID dataset with {num_rows} records...")
    
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
    
    # Create projects (including emergency ones)
    projects = [f"PG-PROJ-{i:03d}" for i in range(1, 26)]
    emergency_projects = [f"PG-PROJ-EMER-{i:03d}" for i in range(1, 6)]
    all_projects = projects + emergency_projects
    
    project_metadata = {}
    for proj in all_projects:
        reg = random.choice(list(region_state_wh.keys()))
        state, wh = random.choice(region_state_wh[reg])
        is_emer = "EMER" in proj
        project_metadata[proj] = {
            "Region": reg,
            "State": state,
            "Warehouse": wh,
            "Budget": random.randint(200000, 600000) if is_emer else random.randint(100000, 400000),
            "Is_Emergency": is_emer
        }
        
    # Commodity Price baseline with inflation (5% annual) and cycles
    commodity_prices = []
    base_price = 100.0
    for idx, dt in enumerate(dates):
        # 5% annual inflation drift
        inflation_mult = 1.0 + (idx / 52) * 0.05
        # 12-month cyclic commodity fluctuation
        cycle_mult = 1.0 + 0.08 * np.sin(2 * np.pi * dt.month / 12)
        random_shock = np.random.normal(0.0, 1.5)
        # 2024 global supply chain crisis (weeks 52 to 80)
        shortage_spike = 1.35 if 52 <= idx <= 80 else 1.0
        
        price = max(70.0, (base_price * cycle_mult * inflation_mult * shortage_spike) + random_shock)
        commodity_prices.append(price)
        
    data_rows = []
    
    # Warehouse & Material combinations
    wh_materials = []
    for reg, locations in region_state_wh.items():
        for state, wh in locations:
            for mat in VALID_MATERIAL_TYPES:
                wh_materials.append((reg, state, wh, mat))
                
    total_combinations = len(wh_materials)
    rows_per_comb = num_rows // total_combinations + 1
    
    row_count = 0
    
    # Track inventory per warehouse-material combination sequentially
    inventory_state = {}
    for reg, state, wh, mat in wh_materials:
        inventory_state[(wh, mat)] = random.randint(1000, 2500)
        
    for reg, state, wh, mat in wh_materials:
        # Determine supplier
        matching_suppliers = [sup for sup, mats in suppliers.items() if mat in mats]
        supplier = random.choice(matching_suppliers) if matching_suppliers else random.choice(supplier_list)
        
        storage_capacity = random.randint(3000, 8000)
        production_capacity = random.randint(2000, 7000)
        
        base_lead_time = {
            "Transformer": 90, "Conductor": 45, "Tower Member": 60,
            "Insulator": 30, "Earthwire": 30, "Hardware Fittings": 20
        }.get(mat, 30)
        
        # Select active projects for this warehouse
        wh_projects = [p for p, meta in project_metadata.items() if meta["Warehouse"] == wh]
        if not wh_projects:
            wh_projects = [random.choice(all_projects)]
            
        current_inventory = inventory_state[(wh, mat)]
        
        for date_idx, dt in enumerate(dates):
            if row_count >= num_rows:
                break
                
            proj = random.choice(wh_projects)
            meta = project_metadata[proj]
            is_emergency = meta["Is_Emergency"]
            proj_budget = meta["Budget"]
            
            # --- operational scenario 1: Season & Weather ---
            month = dt.month
            if month in [12, 1, 2]:
                season = "Winter"
            elif month in [3, 4, 5]:
                season = "Summer"
            elif month in [6, 7, 8, 9]:
                season = "Monsoon"
            else:
                season = "Post-Monsoon"
                
            # Default weather
            weather_probs = [0.75, 0.10, 0.10, 0.02, 0.03]
            # Winter Cold Waves in Northern Region
            if season == "Winter" and reg == "NR":
                weather_probs = [0.40, 0.05, 0.05, 0.50, 0.00]
            # Summer Heat Waves in Western/Northern Regions
            elif season == "Summer" and reg in ["WR", "NR"]:
                weather_probs = [0.40, 0.00, 0.10, 0.00, 0.50]
            # Intense Monsoons in Eastern/North-Eastern Regions
            elif season == "Monsoon" and reg in ["ER", "NER"]:
                weather_probs = [0.15, 0.65, 0.20, 0.00, 0.00]
                
            weather = np.random.choice(
                ["Normal", "Rainy", "Heavy Wind", "Extreme Cold", "Extreme Heat"],
                p=weather_probs
            )
            
            # --- operational scenario 2: Public Holidays & Festivals slowdown ---
            # Construction pauses/slows down in India during Diwali (Oct/Nov) and end-of-year
            is_festival = (month == 11 and (10 <= dt.day <= 20)) or (month == 12 and (25 <= dt.day <= 31))
            festival_mult = 0.55 if is_festival else 1.0
            
            # --- operational scenario 3: Project Acceleration ---
            # 10% chance of unexpected project acceleration
            is_accelerated = np.random.rand() < 0.10
            acceleration_mult = 1.45 if is_accelerated else 1.0
            
            # --- operational scenario 4: Material Shortage Window (Supply chain bottleneck in 2024) ---
            # 2024 global supply chain crunch (weeks 52 to 80) affecting conductors & transformers
            is_shortage_window = (52 <= date_idx <= 80) and (mat in ["Conductor", "Transformer"])
            
            # --- Project Phase sequence & Demand Calculation ---
            phase_idx = min(4, max(0, int((date_idx / len(dates)) * 5)))
            project_phase = VALID_PROJECT_PHASES[phase_idx]
            
            phase_mult = 1.0
            if project_phase == "Foundation" and mat in ["Tower Member", "Hardware Fittings"]:
                phase_mult = 1.6
            elif project_phase == "Tower Erection" and mat == "Tower Member":
                phase_mult = 2.4
            elif project_phase == "Stringing" and mat in ["Conductor", "Insulator", "Earthwire"]:
                phase_mult = 2.8
            elif project_phase == "Testing & Commissioning" and mat == "Transformer":
                phase_mult = 2.1
            elif project_phase == "Planning":
                phase_mult = 0.15
                
            # Monsoon slowdowns for outdoors construction
            monsoon_slowdown = 0.50 if (season == "Monsoon" and reg in ["ER", "NER"]) else 0.85 if (season == "Monsoon") else 1.0
            
            base_demand = {
                "Conductor": 480, "Tower Member": 380, "Insulator": 280,
                "Transformer": 4, "Earthwire": 130, "Hardware Fittings": 180
            }.get(mat, 100)
            
            # Calculate Quantity Required
            demand_trend = 1.0 + (date_idx / len(dates)) * 0.25 # grid demand grows 25% over 3 years
            demand_noise = np.random.normal(0, 0.08 * base_demand)
            
            quantity_required = int(base_demand * phase_mult * monsoon_slowdown * demand_trend * festival_mult * acceleration_mult + demand_noise)
            
            # Emergency projects double material requirements
            if is_emergency:
                quantity_required = int(quantity_required * 2.0)
                
            # Cap quantity_required at positive integer
            quantity_required = max(0, quantity_required)
            
            # Historical Demand (lagged correlation)
            historical_demand = max(0, int(quantity_required * 0.94 + np.random.normal(0, 10)))
            
            # --- operational scenario 5: Inventory Stockouts & Shortage limits ---
            if is_shortage_window:
                # Supply bottleneck empties warehouse inventory
                current_inventory = random.randint(1, 10)
            else:
                # Standard inventory replenishment flow
                if current_inventory < quantity_required:
                    # Place order
                    replenishment = int(quantity_required * 1.6)
                    current_inventory = current_inventory - quantity_required + replenishment
                else:
                    current_inventory = current_inventory - quantity_required
                    
                # Bind inventory within warehouse capacity
                current_inventory = max(0, min(storage_capacity, current_inventory))
                
            inventory_state[(wh, mat)] = current_inventory
            
            # --- operational scenario 6: Lead Time & Supplier Delays ---
            lead_time = base_lead_time
            
            # Monsoon delays affecting shipping routes
            if season == "Monsoon" and reg in ["NER", "ER"]:
                lead_time += random.randint(12, 28)
            # Regional weather transit disruptions
            elif weather in ["Extreme Cold", "Extreme Heat"]:
                lead_time += random.randint(5, 12)
            # Emergency projects request halved lead times (priority expedited)
            if is_emergency:
                lead_time = max(5, int(lead_time * 0.50))
                
            # Random supplier logistical delay (15% probability)
            has_supplier_delay = np.random.rand() < 0.15
            if has_supplier_delay:
                lead_time += random.randint(8, 22)
                
            # Add random variance
            lead_time = max(5, lead_time + random.randint(-4, 6))
            
            # --- operational scenario 7: Dynamic Supplier Risk Score ---
            # Increases with supplier delay, material shortages, and high lead times
            base_risk = 0.10 + (lead_time / 160) * 0.4
            risk_shock = 0.35 if is_shortage_window else 0.15 if has_supplier_delay else 0.0
            supplier_risk = float(np.clip(base_risk + risk_shock + np.random.normal(0, 0.04), 0.05, 0.95))
            
            # --- operational scenario 8: Transportation cost & disruptions ---
            commodity_price = commodity_prices[date_idx]
            
            # Regional distance factor
            distance_factors = {"NR": 1.0, "SR": 1.15, "WR": 1.05, "ER": 1.25, "NER": 1.60}
            dist_factor = distance_factors.get(reg, 1.0)
            
            weather_cost_mult = 1.0
            if weather in ["Rainy", "Heavy Wind"]:
                weather_cost_mult = 1.30
            elif weather in ["Extreme Cold", "Extreme Heat"]:
                weather_cost_mult = 1.15
                
            # Emergency projects pay 2.0x transport cost premium
            emergency_cost_mult = 2.0 if is_emergency else 1.0
            
            # Warehouse Congestion: inventory >85% capacity costs more to handle
            is_congested = current_inventory > (storage_capacity * 0.85)
            congestion_mult = 1.30 if is_congested else 1.0
            
            # Transit disruptions (Strikes 5% chance, Landslides 20% during monsoon in NER)
            is_strike = np.random.rand() < 0.05
            is_landslide = (season == "Monsoon" and reg == "NER" and np.random.rand() < 0.20)
            disruption_mult = 1.60 if (is_strike or is_landslide) else 1.0
            
            # Stockout logistics penalty
            is_stockout = current_inventory < (base_demand * 0.15)
            stockout_mult = 1.50 if is_stockout else 1.0
            
            base_trans_cost = random.randint(20, 50)
            trans_cost = float(
                base_trans_cost 
                * dist_factor 
                * weather_cost_mult 
                * emergency_cost_mult 
                * congestion_mult 
                * disruption_mult 
                * stockout_mult 
                * (commodity_price / 100.0)
            )
            
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
    
    # Apply controlled noise / dirty data (~2% of records) for preprocessing pipeline validation
    logger.info("Applying controlled raw data anomalies (~2% noise) for pipeline validation...")
    
    # 1. Missing values (NaNs)
    cols_to_null = ["Lead_Time_Days", "Historical_Demand", "Current_Inventory", "Weather", "Supplier_Risk"]
    for col in cols_to_null:
        null_indices = df.sample(frac=0.008, random_state=RANDOM_SEED).index
        df.loc[null_indices, col] = np.nan
        
    # 2. Negative values
    neg_inv_indices = df.sample(frac=0.003, random_state=RANDOM_SEED).index
    df.loc[neg_inv_indices, "Current_Inventory"] = -150
    
    neg_demand_indices = df.sample(frac=0.003, random_state=RANDOM_SEED + 1).index
    df.loc[neg_demand_indices, "Quantity_Required"] = -80
    df.loc[neg_demand_indices, "Historical_Demand"] = -80
    
    # 3. Invalid dates
    invalid_date_indices = df.sample(frac=0.004, random_state=RANDOM_SEED + 2).index
    df.loc[invalid_date_indices, "Date"] = "202-INVALID-DATE"
    
    # 4. Invalid categoricals
    invalid_reg_indices = df.sample(frac=0.003, random_state=RANDOM_SEED + 3).index
    df.loc[invalid_reg_indices, "Region"] = "XX"
    
    invalid_phase_indices = df.sample(frac=0.003, random_state=RANDOM_SEED + 4).index
    df.loc[invalid_phase_indices, "Project_Phase"] = "Unknown Phase"
    
    # 5. Duplicate rows (~1.0%)
    dup_indices = df.sample(frac=0.01, random_state=RANDOM_SEED + 5).index
    df_dups = df.loc[dup_indices].copy()
    df = pd.concat([df, df_dups], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    
    logger.info(f"Synthetic dataset generation complete. Dataset shape: {df.shape}")
    return df

def main():
    df = generate_powergrid_data(6000)
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Saved raw dataset to: {RAW_DATA_PATH}")

if __name__ == "__main__":
    main()
