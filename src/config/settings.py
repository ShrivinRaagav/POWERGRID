import os
import yaml
from pathlib import Path

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

# Load YAML Configuration
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found at: {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Resolve directories dynamically from configuration
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Ensure all workspace directories exist
for folder in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# File Paths (mapped from config.yaml relative to BASE_DIR)
RAW_DATA_PATH = BASE_DIR / config["paths"]["raw_data_path"]
PROCESSED_TRAIN_PATH = BASE_DIR / config["paths"]["processed_train_path"]
PROCESSED_VAL_PATH = BASE_DIR / config["paths"]["processed_val_path"]
PROCESSED_TEST_PATH = BASE_DIR / config["paths"]["processed_test_path"]
PROCESSED_DATASET_PATH = BASE_DIR / config["paths"]["processed_dataset_path"]

VALIDATION_REPORT_PATH = BASE_DIR / config["paths"]["validation_report_path"]
FEATURE_SUMMARY_PATH = BASE_DIR / config["paths"]["feature_summary_path"]
DATA_QUALITY_REPORT_PATH = BASE_DIR / config["paths"]["data_quality_report_path"]
DATA_DICTIONARY_PATH = BASE_DIR / config["paths"]["data_dictionary_path"]
PIPELINE_DIAGRAM_PATH = BASE_DIR / config["paths"]["pipeline_diagram_path"]

# Pipeline Parameters
RANDOM_SEED = config["random_seed"]
TRAIN_RATIO = config["split_ratios"]["train"]
VAL_RATIO = config["split_ratios"]["val"]
TEST_RATIO = config["split_ratios"]["test"]

LAG_PERIODS = config["feature_engineering"]["lag_periods"]
ROLLING_WINDOWS = config["feature_engineering"]["rolling_windows"]
FE_EPSILON = float(config["feature_engineering"]["epsilon"])
LEAD_TIME_BINS = config["feature_engineering"]["lead_time_bins"]

# Schema Validation Values
VALID_REGIONS = config["validation"]["allowed_regions"]
VALID_PROJECT_PHASES = config["validation"]["allowed_project_phases"]
VALID_MATERIAL_TYPES = config["validation"]["allowed_material_types"]
DATE_BOUNDS_START = config["validation"]["date_bounds"]["start"]
DATE_BOUNDS_END = config["validation"]["date_bounds"]["end"]

# Logging configurations
LOG_LEVEL = config["logging"]["log_level"]
LOG_FILE_PATH = BASE_DIR / config["logging"]["log_file_path"]
LOG_CONSOLE_OUTPUT = config["logging"]["console_output"]
LOG_FORMAT = config["logging"]["log_format"]

# Core Data Schema Constants (preserved for consistency)
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
