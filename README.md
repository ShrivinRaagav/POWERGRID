# AI-Based Material Demand Forecasting & Supply Chain Optimization for POWERGRID

This project implements a robust, production-ready machine learning data collection, validation, and preprocessing pipeline for material planning in POWERGRID transmission projects. 

The architecture is inspired by the research paper: **"A Machine Learning-Based Approach for Multi-Objective, Multi-Product, and Multi-Period Supply Chain Optimization via Demand Forecasting."**

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Installation & Setup](#installation--setup)
4. [How to Run the Pipeline](#how-to-run-the-pipeline)
5. [Data Validation & Preprocessing Pipeline](#data-validation--preprocessing-pipeline)
6. [Feature Engineering details](#feature-engineering-details)
7. [Train/Validation/Test Split Strategy](#trainvalidationtest-split-strategy)
8. [Roadmap: Integrating Next Phase Components](#roadmap-integrating-next-phase-components)

---

## 🔍 Project Overview

Transmission line material planning (conductors, insulators, towers, transformers, etc.) is a complex task involving multiple regions, warehouses, project phases, and supplier risks. 

This framework implements **Module 1 (Data Collection & Preprocessing)**, establishing a highly structured and scalable foundation. It simulates realistic POWERGRID supply chain dynamics with weekly interval datasets, validates data health, applies modular cleaning, extracts cyclical and lag features, and serializes transformations to disk—avoiding lookahead data leakage.

---

## 📂 Folder Structure

```
POWERGRID/
│
├── data/
│   ├── raw/                  # Raw synthetic dataset with controlled noise
│   ├── processed/            # Scaled, encoded, engineered train/val/test datasets
│   └── external/             # External reference data (empty)
│
├── models/                   # Serialized Python cleaner, encoder, and scaler objects
│
├── reports/                  # Validation reports and feature statistics
│   ├── validation_report.csv
│   └── feature_summary.csv
│
├── src/
│   ├── config/
│   │   └── settings.py       # Pathing, column names, validation schemas
│   ├── data_generation/
│   │   └── generator.py      # Generates 6,000+ rows of synthetic grid planning records
│   ├── preprocessing/
│   │   ├── cleaner.py        # Imputation, outlier handling (IQR), duplicate removal
│   │   ├── encoder.py        # OneHot & Ordinal categoric encoders
│   │   └── scaler.py         # Standard numerical scaling (Z-Score)
│   ├── features/
│   │   ├── temporal.py       # DateTime component extraction and cyclical transforms
│   │   └── engineer.py       # Lags, rolling averages, and custom domain ratios
│   ├── validation/
│   │   └── validator.py      # Quality check assertions and validation logger
│   ├── utils/
│   │   └── helpers.py        # Shared logger configuration
│   └── pipeline.py           # Orchestrator combining steps
│
├── requirements.txt          # Python library dependencies
├── data_dictionary.md        # Description of raw and engineered variables
├── README.md                 # Project documentation
└── run_pipeline.py           # Command-line entry point to execute the pipeline
```

---

## 🛠️ Installation & Setup

1. Ensure Python 3.10+ is installed on your system.
2. Clone this repository or locate the workspace directory.
3. Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run the Pipeline

Execute the full end-to-end data pipeline (generation -> validation -> cleaning -> feature engineering -> scaling/encoding -> reports) using the runner script:

```bash
python run_pipeline.py [encoding_method]
```

- **`encoding_method`** (optional): `ordinal` (default, ideal for tree models) or `onehot` (ideal for SVR/MLP/LSTM).

**Example Output:**
```
==================================================
POWERGRID DATA PREPROCESSING SUMMARY
==================================================
Raw Dataset Shape:           (6060, 22)
Train Dataset Shape (70%):   (4199, 44)
Val Dataset Shape (15%):     (872, 44)
Test Dataset Shape (15%):    (905, 44)
Combined Processed Shape:    (5976, 44)
--------------------------------------------------
Generated Outputs:
1. Processed Dataset:       data/processed/processed_dataset.csv
2. Raw/Clean Validation:    reports/validation_report.csv
3. Feature Summary Statistics: reports/feature_summary.csv
4. Serialized ML Encoders:   models/
==================================================
```

---

## 🧹 Data Validation & Preprocessing Pipeline

To ensure the pipeline is resilient, the data generator injects a small percentage of controlled dirty data (missing values, negatives, duplicates, invalid codes, and out-of-bounds dates).

### Validation Engine (`validator.py`)
Checks for:
- Missing value counts per column.
- Duplicated records.
- Negative values (Inventory and Demand).
- Incorrect Date formats/ranges.
- Out-of-bounds categorical bounds (Region code checks, Project phase strings).

The validator outputs `reports/validation_report.csv`, showing failures in `raw` data, and confirming 100% compliance (`PASS` on all checks) in the `cleaned` data stage.

### Modular Preprocessing (`cleaner.py`, `encoder.py`, `scaler.py`)
1. **Duplicates**: Drops identical rows.
2. **Dates**: Converts strings to Pandas Datetime objects and drops invalid dates.
3. **Categoricals**: Maps invalid codes (e.g. invalid state/region names) and fills missing categorical data using Training Modes.
4. **Numerics**: Fills missing values with Training Medians. Sets negative inventory and demand to zero. Caps outliers using the Interquartile Range (IQR) capping method ($[Q1 - 1.5\text{IQR}, Q3 + 1.5\text{IQR}]$) to prevent model skewing.
5. **Encoding**: Categorical values are ordinal or one-hot encoded and saved to `models/categorical_encoder.joblib`.
6. **Scaling**: Numerical features are standardized via Z-score scaling and saved to `models/numerical_scaler.joblib`.

---

## 📈 Feature Engineering Details

Module 1 creates 22 additional engineered features to feed predictive algorithms in the next phase.

### Temporal & Cyclical Features
- Extracts standard fields: `Year`, `Month`, `WeekOfYear`, `DayOfWeek`, and fiscal `Is_Quarter_End`.
- Converts Month and Week into **Cyclical coordinates** using Sine and Cosine transformations (e.g., $Month\_Sin = \sin(2\pi \cdot Month / 12)$). This preserves chronological proximity (e.g., December and January are recognized as adjacent).

### Lag & Rolling Averages
To predict demand, time-series features are calculated by grouping by `[Warehouse, Material_Type]` and sorting by `Date`:
- **Lags**: `Lag_1`, `Lag_2`, and `Lag_3` represent prior demands.
- **Rolling Means**: `Rolling_Mean_3` and `Rolling_Mean_6` provide smoothed demand baselines.
*Note: Shifting by 1 period is applied before computing rolling averages to prevent lookahead target leakage.*

### Supply Chain Domain Features
- **Inventory Utilization**: `Current_Inventory / Storage_Capacity` (utilization rate).
- **Lead Time Category**: Binned `Lead_Time_Days` into Short (0), Medium (1), or Long (2).
- **Demand Growth Rate**: `(Lag_1 - Lag_2) / (Lag_2 + epsilon)`.
- **Inventory Coverage**: `Current_Inventory / Rolling_Mean_3` (how many periods current warehouse stock will cover).
- **Budget Utilization**: Value of lagged demand compared to allocated project budget.
- **Supplier Risk Score**: Combines `Supplier_Risk` rating and `Lead_Time_Days` multiplier.
- **Seasonal Demand Index**: Calculated as:
  $$\text{Seasonal Index} = \frac{\text{Average seasonal demand for material}}{\text{Overall average demand for material}}$$
  This maps the relative seasonality influence for each material.
- **Transportation Cost Index**: `Transportation_Cost / Commodity_Price`.

---

## ⏳ Train/Validation/Test Split Strategy

For time-series forecasting, standard random cross-validation causes **data leakage** (lookahead bias). 

We implement a **Chronological Split** based on dates:
- **Training Set (70%)**: Earliest chronological periods. All cleaners, feature engineers, encoders, and scalers are **fit only on this partition**.
- **Validation Set (15%)**: Middle chronological periods. Transformed using parameters fit on Training.
- **Test Set (15%)**: Final chronological periods. Transformed using parameters fit on Training.

This split guarantees that the model is validated on future timeframes relative to its training set, representing real production performance.

---

## 🗺️ Roadmap: Integrating Next Phase Components

Module 1's architecture is explicitly designed to integrate the subsequent layers of the research paper:

```
                  ┌──────────────────────────────┐
                  │   Processed Data Splits      │
                  │   (train, val, test CSVs)    │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │ Time Series Decomposition    │
                  │ (apply EMD & DWT per group)  │
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │   ML Forecasting Models      │
                  │ (LSTM, XGBoost, LightGBM, RF)│
                  └──────────────┬───────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────┐
                  │  Multi-Objective Supply      │
                  │     Chain Optimization       │
                  │ (PuLP / CBC Linear Program)  │
                  └──────────────────────────────┘
```

1. **EMD & DWT Decomposition**: 
   Since datasets are saved sequentially at the `[Warehouse, Material_Type]` level, the next phase can easily partition the processed arrays and apply Empirical Mode Decomposition (EMD) or Discrete Wavelet Transform (DWT) to split demand series into trend and high-frequency components.
2. **Forecasting Models**:
   Cleaned datasets containing temporal, cyclical, lag, and domain features can be fed straight into regressors like Random Forest, XGBoost, SVR, or Multilayer Perceptrons (MLP). The sequential splits are ready for Long Short-Term Memory (LSTM) recurrent network shape formatting.
3. **Multi-Objective Linear Optimization**:
   The predicted demand outputs, coupled with the engineered supplier risk, budgets, transportation costs, and capacities, can be passed directly to the `PuLP` library. There, a linear programming model will minimize supply chain cost and supplier risk under constraints (store capacity, inventory balance, delivery time, etc.) using the CBC Solver.
