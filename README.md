# AI-Based Material Demand Forecasting & Supply Chain Optimization for POWERGRID

This project implements a robust, production-ready machine learning data collection, validation, and preprocessing pipeline for material planning in POWERGRID transmission projects. It is designed to satisfy the requirements of the Smart India Hackathon problem statement.

The project architecture is inspired by the research paper: **"A Machine Learning-Based Approach for Multi-Objective, Multi-Product, and Multi-Period Supply Chain Optimization via Demand Forecasting."**

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Research Motivation](#2-research-motivation)
3. [POWERGRID Problem Statement](#3-powergrid-problem-statement)
4. [Current Progress (30% Scope)](#4-current-progress-30-scope)
5. [Completed Modules](#5-completed-modules)
6. [Architecture Diagram](#6-architecture-diagram)
7. [Pipeline Diagram](#7-pipeline-diagram)
8. [Folder Structure](#8-folder-structure)
9. [Execution Instructions](#9-execution-instructions)
10. [Generated Reports](#10-generated-reports)
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

## 4. Current Progress (30% Scope)
This framework strictly covers **Module 1 (Data Collection & Preprocessing)**:
*   **Operational Simulator**: Generates 6,000+ sequential records modeling monsoons, weather events, supplier delays, emergency projects, material shortages, inflation, holiday slowdowns, warehouse congestion, and transit strikes.
*   **Centralized Configuration**: All variables are managed via [config/config.yaml](file:///c:/Users/kavsh/Desktop/POWERGRID/config/config.yaml).
*   **Feature Stabilization**: Implements mathematically stable and capped equations for `Demand_Growth` and `Inventory_Coverage`.
*   **Outlier & Missingness Treatment**: Imputation and standard IQR outlier Winsorization fit strictly on training splits to prevent data leakage.
*   **Chronological Split**: Enforces sequential date splitting to prevent target leaks during future forecasting.
*   **Automated Research Outputs**: Automatically compiles three research reports at runtime: Data Quality, Data Dictionary, and Pipeline Diagram.

---

## 5. Completed Modules
*   **Configuration Manager**: [settings.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/config/settings.py) dynamically reads environment constraints from YAML.
*   **Operational Simulator**: [generator.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/data_generation/generator.py) generates operational scenario datasets.
*   **Validation Suite**: [validator.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/validation/validator.py) tests ranges, categories, duplicates, and missing values.
*   **Modular Preprocessing**: [cleaner.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/preprocessing/cleaner.py), [encoder.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/preprocessing/encoder.py), and [scaler.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/preprocessing/scaler.py) handle duplicates, imputations, capping, encoding, and scaling.
*   **Feature Extraction**: [temporal.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/features/temporal.py) and [engineer.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/features/engineer.py) compute temporal and supply chain metrics.
*   **Orchestrator Pipeline**: [pipeline.py](file:///c:/Users/kavsh/Desktop/POWERGRID/src/pipeline.py) manages the sequence.

---

## 6. Architecture Diagram

```
c:\Users\kavsh\Desktop\POWERGRID\
├── config/
│   └── config.yaml           # Centralized configuration settings
├── logs/
│   └── pipeline.log          # Detailed execution logs
├── reports/                  # Automatically generated MD and CSV reports
├── src/                      # Source code directories
│   ├── config/
│   ├── data_generation/
│   ├── preprocessing/
│   ├── features/
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
Feature Summary
        │
        ▼
Processed Dataset
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

### 2. Run the Data Preprocessing Pipeline
```bash
python run_pipeline.py [ordinal|onehot]
```
*   `ordinal` (default): Optimal for gradient boosted tree models (XGBoost, RandomForest).
*   `onehot`: Optimal for neural networks (MLP, LSTM) and SVR models.

### 3. Run Automated Tests
To run the automated validation tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 10. Generated Reports
Executing the pipeline dynamically updates:
*   [reports/data_quality_report.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/data_quality_report.md): Validates raw vs cleaned values, nulls, duplicates, and outlier caps.
*   [reports/data_dictionary.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/data_dictionary.md): Describes all 44 fields in detail.
*   [reports/pipeline_diagram.md](file:///c:/Users/kavsh/Desktop/POWERGRID/reports/pipeline_diagram.md): Displays pipeline flowcharts.

---

## 11. Future Work
The completed Module 1 features are structured to feed future research modules:
1.  **Decomposition (EMD & DWT)**: Apply EMD and DWT to clean lag sequences.
2.  **ML Forecasting**: Train MLP, SVR, RandomForest, XGBoost, and LSTM algorithms on preprocessed variables.
3.  **Multi-Objective LP Solver**: Supply predicted demands and risk scores into a `PuLP` solver to optimize procurement, inventory levels, and freight routes.
