# AI-Based Material Demand Forecasting & Supply Chain Optimization for POWERGRID

This project implements a robust, production-ready machine learning data collection, validation, and preprocessing pipeline for material planning in POWERGRID transmission projects. It is designed to satisfy the requirements of the Smart India Hackathon problem statement.

The project architecture is inspired by the research paper: **"A Machine Learning-Based Approach for Multi-Objective, Multi-Product, and Multi-Period Supply Chain Optimization via Demand Forecasting."**

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Research Motivation](#research-motivation)
3. [Architecture Diagram](#architecture-diagram)
4. [Folder Structure](#folder-structure)
5. [Current Progress (30% Scope)](#current-progress-30-scope)
6. [Future Modules (70% Roadmap)](#future-modules-70-roadmap)
7. [Installation & Setup](#installation--setup)
8. [Execution Guide](#execution-guide)
9. [Expected Deliverables](#expected-deliverables)

---

## 🔍 Project Overview

Transmission grid expansion projects (towers, substations, conductors) face severe supply chain bottlenecks, transport delays, and budget constraints. This framework implements **Module 1 (Data Collection, Cleaning, Preprocessing, Validation, and Feature Engineering)**.

It simulates a highly realistic operational grid environment, validates data quality using a custom verification suite, performs modular data cleaning, engineers cyclical and time-lagged parameters, and normalizes inputs—laying the foundation for future decomposition and multi-objective optimization solvers.

---

## 💡 Research Motivation

Based on the referenced IEEE publication, traditional grid planning models rely on static historical demand baselines, exposing supply chains to stockouts, inventory congestion, and supplier delay penalties. 

This research establishes a framework where:
1. **Dynamic Forecasting** replaces static averages by incorporating multi-period lag profiles, seasonal variables, and project stage parameters.
2. **Decomposition Techniques** (DWT/EMD) decompose demand series to separate short-term weather/seasonal logistics spikes from long-term project trend components.
3. **Multi-Objective Linear Optimization** balances procurement costs, storage overheads, delivery durations, and supplier risk factors to generate optimal replenishment policies under warehouse and budget limits.

---

## 📊 Architecture Diagram

The visual flow of the data preprocessing and feature engineering sequence is structured as follows:

```
                  ┌──────────────────────────────┐
                  │         Raw Dataset          │
                  │      (raw_dataset.csv)       │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     Pre-Cleaning Check       │
                  │    (validator.py log)        │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │        Data Cleaning         │
                  │   (imputers, outlier caps)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     Feature Engineering      │
                  │  (cyclical time, lags, SC)   │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    Post-Cleaning Validate    │
                  │     (zero errors logged)     │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │    Chronological Splitting   │
                  │   (70% Train/15% Val/15% Test)│
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │     Encoding & Scaling       │
                  │  (StandardScaler/Ordinalfit) │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │      Processed Outputs       │
                  │   (train, val, test CSVs)    │
                  └──────────────────────────────┘
```

For a detailed view of the node pipeline, refer to [reports/pipeline_diagram.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/pipeline_diagram.md).

---

## 📂 Folder Structure

```
POWERGRID/
│
├── config/
│   └── config.yaml           # Centralized configuration variables (paths, splits, seeds)
│
├── data/
│   ├── raw/                  # Raw synthetic dataset with controlled noise
│   ├── processed/            # Scaled, encoded, engineered train/val/test datasets
│   └── external/             # External reference data (empty)
│
├── logs/                     # Persistent log files recording pipeline runs
│   └── pipeline.log
│
├── models/                   # Serialized Python cleaner, encoder, and scaler objects
│
├── reports/                  # Validation reports, summaries, and visualizations
│   ├── data_quality_report.md
│   ├── data_dictionary.md
│   ├── pipeline_diagram.md
│   ├── validation_report.csv
│   └── feature_summary.csv
│
├── src/
│   ├── config/
│   │   └── settings.py       # Reads YAML variables and exposes constants
│   ├── data_generation/
│   │   └── generator.py      # Generates operational scenario grid datasets (6,000+ rows)
│   ├── preprocessing/
│   │   ├── cleaner.py        # Imputation, outlier capping, duplicate removal
│   │   ├── encoder.py        # Categoric encoders (Ordinal/One-Hot)
│   │   └── scaler.py         # Z-Score numerical standardizer
│   ├── features/
│   │   ├── temporal.py       # Cyclical time transforms (sine/cosine)
│   │   └── engineer.py       # Chronological lags, rolling averages, supply chain metrics
│   ├── validation/
│   │   └── validator.py      # Asserts data constraints and quality criteria
│   └── utils/
│       ├── helpers.py        # Dual logger setup (Console + File)
│       └── reports_generator.py # Automatically builds research reports at runtime
│
├── tests/                    # Automated testing suite
│   └── test_pipeline.py
│
├── requirements.txt          # Library dependencies
├── README.md                 # Project README
└── run_pipeline.py           # Orchestrator CLI entry point
```

---

## 📈 Current Progress (30% Scope)

This framework strictly covers **Module 1 (Data Collection & Preprocessing)**:
*   **Operational Simulator**: Generates 6,000+ sequential records modeling:
    *   *Monsoon impacts* (extended lead times and transport cost overheads in rain-heavy regions).
    *   *Weather anomalies* (heat/cold wave logistics disruptions and supplier risk shocks).
    *   *Emergency projects* (doubled demand, priority expedited shipping, and cost premiums).
    *   *Supply bottlenecks* (materials shortages driving inventories down and raw commodity prices up).
    *   *Logistics penalties* (warehouse congestion costs, stockout shipping premiums, strikes/landslides).
*   **Centralized Configuration**: All variables (paths, validation constraints, split ratios, seeds, parameters) are managed via [config/config.yaml](file:///c:/Users/kavsh/Desktop/POWERGRID/config/config.yaml).
*   **Validation Suite**: Automatic quality checking (pre-cleaning vs. post-cleaning tests).
*   **Outlier & Missingness Treatment**: Imputation fit on the training split only and standard IQR outlier Winsorization.
*   **Feature Engineering**: Creates 22 engineered features, including cyclical dates, chronological warehouse-material lagged demands (1-3 weeks), rolling averages (3 & 6 weeks), inventory utilization, lead time categories, risk scores, and seasonal indices.
*   **Chronological Split**: Enforces sequential date splitting to prevent target leaks during future forecasting.
*   **Automated Research Outputs**: Automatically compiles three research reports at runtime: Data Quality, Data Dictionary, and Pipeline Mermaid flow.

---

## 🛣️ Future Modules (70% Roadmap)

To maintain strict scope, the following modules are **NOT** implemented in this phase, but are supported by Module 1's sequential preprocessing design:

1.  **Decomposition Layer**: Parse `train_dataset.csv` sequences at the `[Warehouse, Material_Type]` level and apply Empirical Mode Decomposition (EMD) and Discrete Wavelet Transform (DWT) to resolve high-frequency supply noises.
2.  **ML Forecasting**: Supply the engineered lags, cyclical dates, and season indices into regressors (MLP, LSTM, SVR, RandomForest, XGBoost, and LightGBM Quantile Regression) to predict future material requirements.
3.  **Multi-Objective LP Optimization**: Feed the forecasted demand, supplier risk scores, and transportation costs into a `PuLP` solver using the CBC solver to optimize procurement, inventory levels, and logistics routes under constraints.

---

## 🛠️ Installation & Setup

1. Check that Python 3.10+ is installed:
   ```bash
   python --version
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Execution Guide

### 1. Run the Data Pipeline
To execute the pipeline, run the root runner script. You can specify the categorical encoding method (`ordinal` or `onehot`):

```bash
python run_pipeline.py [ordinal|onehot]
```

This will run the simulation, cleaning, feature engineering, and automatically generate all research CSVs and Markdown reports.

### 2. Run Automated Unit Tests
To run the validation test suite (verifying config loading, generator, cleaner, validator, and features):

```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 💾 Expected Deliverables

Executing the pipeline outputs the following files:

*   **Processed Train/Val/Test Datasets**: `data/processed/train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`
*   **Combined Dataset**: `data/processed/processed_dataset.csv`
*   **Pipeline Log file**: `logs/pipeline.log`
*   **Validation Check Log**: `reports/validation_report.csv` (pre- vs. post-cleaning logs)
*   **Features Statistics**: `reports/feature_summary.csv`
*   **Data Quality Report**: [reports/data_quality_report.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/data_quality_report.md)
*   **Data Dictionary**: [reports/data_dictionary.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/data_dictionary.md)
*   **Pipeline Mermaid Diagram**: [reports/pipeline_diagram.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/pipeline_diagram.md)
*   **Serialized ML Objects**: `models/data_cleaner.joblib`, `feature_engineer.joblib`, `categorical_encoder.joblib`, `numerical_scaler.joblib`
