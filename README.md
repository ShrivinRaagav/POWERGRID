# AI-Based Material Demand Forecasting & Supply Chain Optimization for POWERGRID

This project implements a robust, production-ready machine learning data collection, validation, and preprocessing pipeline for material planning in POWERGRID transmission projects. It is designed to satisfy the requirements of the Smart India Hackathon problem statement.

The project architecture is inspired by the research paper: **"A Machine Learning-Based Approach for Multi-Objective, Multi-Product, and Multi-Period Supply Chain Optimization via Demand Forecasting."**

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Research Motivation](#2-research-motivation)
3. [POWERGRID Problem Statement](#3-powergrid-problem-statement)
4. [Current Progress (55% Scope)](#4-current-progress-55-scope)
5. [Completed Modules](#5-completed-modules)
6. [Architecture Diagram](#6-architecture-diagram)
7. [Pipeline Diagram](#7-pipeline-diagram)
8. [Folder Structure](#8-folder-structure)
9. [Execution Instructions](#9-execution-instructions)
10. [Generated Reports and Artifacts](#10-generated-reports-and-artifacts)
11. [Future Work](#11-future-work)

---

## 1. Project Overview
Transmission grid infrastructure projects (conductors, insulators, towers, transformers, etc.) require precise procurement schedules to prevent project cost overruns and delays. This framework implements **Module 1 (Data Collection, Cleaning, Preprocessing, Validation, and Feature Engineering)**. 

It simulates a highly realistic operational grid environment, validates data quality using a custom verification suite, performs modular data cleaning, engineers cyclical and time-lagged parameters, and normalizes inputs—laying the foundation for future decomposition and multi-objective optimization solvers.

---

## 2. Research Motivation
Based on the referenced IEEE publication, traditional grid planning models rely on static historical demand baselines, exposing supply chains to stockouts, inventory congestion, and supplier delay penalties. 

This research establishes a framework where:
1. **Dynamic Forecasting** replaces static averages by incorporating multi-period lag profiles, seasonal variables, and project stage parameters.
2. **Decomposition Techniques** (DWT/EMD) decompose demand series to separate short-term weather/seasonal logistics spikes from long-term project trend components.
3. **Multi-Objective Linear Optimization** balances procurement costs, storage overheads, delivery durations, and supplier risk factors to generate optimal replenishment policies under warehouse and budget limits.

---

## 3. POWERGRID Problem Statement
Under the Smart India Hackathon scope, POWERGRID (Ministry of Power) requires an intelligent, data-driven material planning system. Transmission lines span thousands of kilometers and pass through extreme geographical regions. Logistics are vulnerable to weather disruptions, regional monsoon delays, supplier risk variations, and sudden project accelerations. 

This module sets up a robust data pipeline that simulates these operational realisms, cleans input errors, handles outliers, and computes mathematical features suitable for machine learning forecasting.

---

## 4. Current Progress (55% Scope)
This framework covers **Module 1 (Data Collection & Preprocessing)** and **Module 2 (Time-Series Analysis & Decomposition Refinement)**:
*   **Operational Simulator**: Generates 6,000+ sequential records modeling monsoons, weather delays, emergencies, material shortages, and labor events.
*   **Centralized Configuration**: All variables are managed via [config/config.yaml](file:///c:/Users/kavsh/Desktop/POWERGRID/config/config.yaml).
*   **Feature Stabilization**: Implements mathematically stable and capped equations for `Demand_Growth` and `Inventory_Coverage`.
*   **Time-Series Preparation**: Formats processed panel data into sorted chronological sequences and handles missing weekly timestamps using configurable methods (zero fill, ffill, linear interpolation).
*   **Discrete Wavelet Transform (DWT)**: Performs multi-level wavelet decomposition (default `db4`) to extract approximation and detail coefficients.
*   **Empirical Mode Decomposition (EMD)**: Resolves non-stationary patterns into Intrinsic Mode Functions (IMFs) and residual trends.
*   **Reconstruction Validation**: Computes DWT and EMD signal reconstruction RMSE, MAE, and correlation per group to ensure zero-loss representation.
*   **Feature Selection & Dimensionality Reduction**: Automatically removes duplicate, constant, low-variance, and highly correlated features (keeping the one with higher Mutual Information importance). Compiles [reports/feature_selection_report.csv](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/feature_selection_report.csv), reducing dataset columns from ~85 to 24.
*   **Model Evaluation Framework**: Provides reusable point and interval forecast evaluation utilities (MAE, RMSE, MAPE, R², SMAPE, and Pinball Loss) and comparison/plotting registries for forecasting in Module 3.

## 5. Completed Modules
*   **Configuration Manager**: [settings.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/config/settings.py) dynamically reads environment constraints from YAML.
*   **Operational Simulator**: [generator.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/data_generation/generator.py) generates operational scenario datasets.
*   **Validation Suite**: [validator.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/validation/validator.py) tests ranges, categories, duplicates, and missing values.
*   **Modular Preprocessing**: [cleaner.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/preprocessing/cleaner.py), [encoder.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/preprocessing/encoder.py), and [scaler.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/preprocessing/scaler.py) handle duplicates, imputations, capping, encoding, and scaling.
*   **Feature Engineering**: [temporal.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/features/temporal.py) and [engineer.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/features/engineer.py) compute temporal and supply chain metrics.
*   **Time-Series Utilities**: [utils.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/time_series/utils.py) structures, groups, and imputes weekly panel datasets.
*   **Wavelet Transform**: [dwt.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/time_series/dwt.py) decomposes signal into multi-level approximation and detail coefficients.
*   **Empirical Mode Decomposition**: [emd.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/time_series/emd.py) extracts IMFs and residual trends.
*   **Decomposition Orchestrator**: [decomposition.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/time_series/decomposition.py) performs classical, DWT, and EMD analysis, and extracts advanced signal features.
*   **Reconstruction Validator**: [reconstruction_validator.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/time_series/reconstruction_validator.py) computes reconstruction statistics (RMSE, MAE, correlation).
*   **Feature Selection Module**: `src/feature_selection/` ([feature_selector.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/feature_selection/feature_selector.py), [variance.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/feature_selection/variance.py), [correlation.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/feature_selection/correlation.py), [importance.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/feature_selection/importance.py), [utils.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/feature_selection/utils.py)) identifies and removes redundant variables.
*   **Evaluation Framework**: `src/evaluation/` ([metrics.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/evaluation/metrics.py), [comparison.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/evaluation/comparison.py), [visualization.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/evaluation/visualization.py)) implements reusable forecast scoring metrics and charts.
*   **Forecasting Abstract Model Interface**: [base_model.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/models/base_model.py) defines the standard interface for ML models.
*   **Model Registry**: [registry.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/models/registry.py) dynamically tracks and accesses registered forecasting models.
*   **Experiment Manager**: [experiment_manager.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/experiments/experiment_manager.py) structures output subdirectories.
*   **Experiment Logger**: [experiment_logger.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/experiments/experiment_logger.py) appends run metrics to central CSV tables.
*   **Training CLI Controller**: [train.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/models/train.py) manages training, prediction, evaluation, and logging pipelines.
*   **Visualization Engine**: [visualization.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/time_series/visualization.py) plots signal components, heatmaps, importances, and saves them as PNG.
*   **Orchestrator Pipeline**: [pipeline.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/pipeline.py) manages the sequence.

---

## 6. Architecture Diagram

```
c:\Users\kavsh\Desktop\POWERGRID\
├── artifacts/                # Serialized estimators and selector objects (.joblib)
├── config/
│   └── config.yaml           # Centralized configuration settings
├── experiments/              # central experiment log and individual run directories
│   ├── checkpoints/          # Model states checkpoints (.joblib)
│   ├── results/              # Prediction outputs CSVs
│   ├── plots/                # Plot visualizations (PNG)
│   ├── logs/                 # Run-specific execution logs
│   └── model_results.csv     # Central experiment tracking log
├── reports/                  # Markdown & CSV analysis reports
│   ├── important_plots/      # Correlation heatmaps and representative decompositions
│   ├── dataset_version.json  # Unique dynamic JSON version of current features
│   └── pipeline_version.md   # Complete pipeline documentation and changelogs
├── src/                      # Source code directories
│   ├── config/
│   ├── data_generation/
│   ├── preprocessing/
│   ├── features/
│   ├── time_series/          # DWT, EMD, and signal feature extraction modules
│   ├── feature_selection/    # Variance, correlation, and Mutual Info selector modules
│   ├── evaluation/           # Reusable model metrics and visualization utilities
│   ├── experiments/          # Logger and directories managers
│   ├── models/               # Abstract base model, registry, mock model, training CLI
│   ├── validation/
│   └── utils/
├── tests/                    # Automated unit tests suite
└── run_pipeline.py           # Pipeline runner entrypoint
```

---

## 7. Pipeline Diagram

```
POWERGRID Dataset
        │
        ▼
Data Validation
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Time-Series Analysis & Decomposition
        │
        ▼
Feature Selection & Dimensionality Reduction
        │
        ▼
Enriched & Selected Processed Dataset (24 columns)
```

For the detailed layout including train/val/test splits, see [reports/pipeline_diagram.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/pipeline_diagram.md).

---

## 8. Folder Structure
For a detailed tree diagram of files, see [Architecture Diagram](#6-architecture-diagram).

---

## 9. Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Data Preprocessing & Selection Pipeline
```bash
python run_pipeline.py [ordinal|onehot]
```
*   `ordinal` (default): Optimal for gradient boosted tree models (XGBoost, RandomForest).
*   `onehot`: Optimal for neural networks (MLP, LSTM) and SVR models.

### 3. Run the CLI Model Training Controller (Module 3)
Run any implemented/registered forecasting model. For example, to run the verification dry-run flow with `mock_model`:
```bash
python -m src.models.train --model mock_model --notes "Dry-run verification of model registry and logger"
```
Future forecasting models (e.g. `random_forest`, `xgboost`, `lstm`) will be executed via the exact same syntax:
```bash
python -m src.models.train --model xgboost
```

### 4. Run Automated Tests
To run the automated validation, feature selection, and model interface tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 10. Generated Reports and Artifacts
Executing the pipeline dynamically updates:
*   [reports/data_quality_report.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/data_quality_report.md): Validates raw vs cleaned values, nulls, duplicates, and outlier caps.
*   [reports/data_dictionary.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/data_dictionary.md): Describes raw fields in detail.
*   [reports/feature_catalog.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/feature_catalog.md): Describes all 78+ generated columns, origins, formulas, and active forecasting status.
*   [reports/decomposition_quality_report.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/decomposition_quality_report.md): Validates DWT/EMD parameters and reconstruction errors.
*   [reports/feature_selection_report.csv](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/feature_selection_report.csv): Tabulates selection drop reasons.
*   [reports/reconstruction_report.csv](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/reconstruction_report.csv): Quantifies transform RMSE, MAE, and correlation.
*   [reports/important_plots/](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/important_plots/): Representative decomposition plots, correlation heatmaps, and feature importances.
*   [reports/dataset_version.json](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/dataset_version.json): Stores active version hash, sample counts, and feature arrays.
*   [reports/pipeline_version.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/pipeline_version.md): Documents the modules sequence, directory layout, and release history.
*   [experiments/model_results.csv](file:///c:/Users/kavsh/Desktop/POWERGRID/experiments/model_results.csv): central CSV recording timestamped run metrics (MAE, RMSE, SMAPE, etc.) and git revisions.
*   [artifacts/](file:///c:/Users/kavsh/Desktop/POWERGRID/artifacts/): Serialized DWT, EMD, and feature selector joblib objects.

---

## 11. Future Work
The completed modules lay a robust foundation for the final steps:
1.  **ML Forecasting**: Register and train forecasting algorithms (`random_forest`, `xgboost`, `lightgbm`, `lstm`, `mlp`, `svr`, `prophet`) using the CLI controller `src/models/train.py`, evaluating point and interval predictions using the `src/evaluation/` metrics.
2.  **Multi-Objective LP Solver**: Supply predicted demands, risk profiles, and costs into a `PuLP`/`CBC` linear programming solver to optimize multi-period procurement and routing under capacity and budget constraints.
