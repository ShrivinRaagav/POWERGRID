import os
from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "raw_dataset.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_dataset.csv"
PROCESSED_VAL_PATH = PROCESSED_DATA_DIR / "val_dataset.csv"
PROCESSED_TEST_PATH = PROCESSED_DATA_DIR / "test_dataset.csv"

VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.csv"
FEATURE_SUMMARY_PATH = REPORTS_DIR / "feature_summary.csv"
DATA_DICTIONARY_PATH = BASE_DIR / "data_dictionary.md"

# Columns Definition
DATE_COL = "Date"
TARGET_COL = "Quantity_Required"

CATEGORICAL_COLS = [
    "Project_ID",
    "Region",
    "State",
    "Warehouse",
    "Supplier",
    "Material_Type",
    "Project_Phase",
    "Tower_Type",
    "Substation_Type",
    "Weather",
    "Season"
]

NUMERICAL_COLS = [
    "Historical_Demand",
    "Current_Inventory",
    "Lead_Time_Days",
    "Supplier_Risk",
    "Commodity_Price",
    "Transportation_Cost",
    "Storage_Capacity",
    "Production_Capacity",
    "Project_Budget"
]

# Validation Schemas
VALID_REGIONS = ["NR", "ER", "WR", "SR", "NER"]
VALID_PROJECT_PHASES = [
    "Planning",
    "Foundation",
    "Tower Erection",
    "Stringing",
    "Testing & Commissioning"
]
VALID_MATERIAL_TYPES = [
    "Conductor",
    "Tower Member",
    "Insulator",
    "Transformer",
    "Earthwire",
    "Hardware Fittings"
]

# Seed for reproducibility
RANDOM_SEED = 42
