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

# =====================================================================
# MODULE 2: TIME-SERIES ANALYSIS PARAMETERS
# =====================================================================
TS_CONFIG = config.get("time_series", {})
TS_GROUP_BY = TS_CONFIG.get("group_by", ["Material_Type", "Region"])
TS_DATE_COL = TS_CONFIG.get("date_col", "Date")
TS_TARGET_COL = TS_CONFIG.get("target_col", "Quantity_Required")
TS_FREQ = TS_CONFIG.get("freq", "W-SUN")
TS_FILL_METHOD = TS_CONFIG.get("fill_method", "zero")

# DWT settings
TS_DWT_WAVELET = TS_CONFIG.get("dwt", {}).get("wavelet", "db4")
TS_DWT_LEVEL = int(TS_CONFIG.get("dwt", {}).get("level", 3))
TS_DWT_OUTPUT_PATH = BASE_DIR / TS_CONFIG.get("dwt", {}).get("output_path", "artifacts/dwt_transform.joblib")

# EMD settings
TS_EMD_SPLINE = TS_CONFIG.get("emd", {}).get("spline_kind", "cubic")
TS_EMD_MAX_IMFS = int(TS_CONFIG.get("emd", {}).get("max_imfs", 5))
TS_EMD_STD_THR = float(TS_CONFIG.get("emd", {}).get("std_thr", 0.2))
TS_EMD_ENERGY_RATIO_THR = float(TS_CONFIG.get("emd", {}).get("energy_ratio_thr", 0.2))
TS_EMD_OUTPUT_PATH = BASE_DIR / TS_CONFIG.get("emd", {}).get("output_path", "artifacts/emd_processor.joblib")

# Features & Output settings
TS_FEATURES_OUTPUT_PATH = BASE_DIR / TS_CONFIG.get("features", {}).get("output_path", "artifacts/feature_extractor.joblib")
TS_SUMMARY_PATH = BASE_DIR / TS_CONFIG.get("features", {}).get("summary_path", "reports/decomposition_summary.csv")

# Visualization settings
TS_PLOTS_DIR = BASE_DIR / TS_CONFIG.get("visualization", {}).get("plots_dir", "reports/important_plots/")
TS_PLOTS_DPI = int(TS_CONFIG.get("visualization", {}).get("dpi", 150))

# Ensure artifacts and visualization directories exist
TS_DWT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
TS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# MODULE 2 REFINEMENT: FEATURE SELECTION & EVALUATION PARAMETERS
# =====================================================================
FS_CONFIG = config.get("feature_selection", {})
FS_VARIANCE_THR = float(FS_CONFIG.get("variance_threshold", 0.01))
FS_CORRELATION_THR = float(FS_CONFIG.get("correlation_threshold", 0.85))
FS_IMPORTANCE_METHOD = FS_CONFIG.get("importance_method", "mutual_info")
FS_TOP_K = int(FS_CONFIG.get("top_k_features", 20))
FS_KEEP_COLS = FS_CONFIG.get("keep_columns", ["Date", "Project_ID", "Region", "Material_Type", "Quantity_Required"])

FS_SELECTOR_PATH = BASE_DIR / FS_CONFIG.get("selector_output_path", "artifacts/feature_selector.joblib")
FS_CORRELATION_PATH = BASE_DIR / FS_CONFIG.get("correlation_output_path", "artifacts/correlation_filter.joblib")
FS_REPORT_PATH = BASE_DIR / FS_CONFIG.get("report_path", "reports/feature_selection_report.csv")
FS_RECONSTRUCTION_REPORT_PATH = BASE_DIR / FS_CONFIG.get("reconstruction_report_path", "reports/reconstruction_report.csv")
FS_QUALITY_REPORT_PATH = BASE_DIR / FS_CONFIG.get("quality_report_path", "reports/decomposition_quality_report.md")
FS_CATALOG_PATH = BASE_DIR / FS_CONFIG.get("catalog_path", "reports/feature_catalog.md")

# Ensure directories for outputs exist
FS_SELECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
FS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =====================================================================
# FORECASTING MODEL INFRASTRUCTURE PARAMETERS
# =====================================================================
FC_CONFIG = config.get("forecasting", {})
RESULTS_CSV_PATH = BASE_DIR / FC_CONFIG.get("results_csv_path", "experiments/model_results.csv")
DATASET_VERSION_PATH = BASE_DIR / FC_CONFIG.get("dataset_version_path", "reports/dataset_version.json")
PIPELINE_VERSION_PATH = BASE_DIR / FC_CONFIG.get("pipeline_version_path", "reports/pipeline_version.md")
EXPERIMENTS_DIR = BASE_DIR / FC_CONFIG.get("experiments_dir", "experiments/")
FORECAST_RANDOM_SEED = int(FC_CONFIG.get("random_seed", 42))

MODELS_CONFIG = config.get("models", {})

# Ensure directories for forecasting results exist
RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
# Create experiment subfolders: results, plots, logs, checkpoints
for folder in ["results", "plots", "logs", "checkpoints"]:
    (EXPERIMENTS_DIR / folder).mkdir(parents=True, exist_ok=True)



