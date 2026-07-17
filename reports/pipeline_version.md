# POWERGRID ML Pipeline Versioning Report

This report documents the current architecture, components, and version history of the POWERGRID material demand forecasting and supply chain optimization data pipeline.

---

## 🛠️ Pipeline Modules and Sequence

The pipeline executes sequentially in a fully automated workflow, passing data from raw simulation through cleaning, feature engineering, signal decomposition, and mathematical feature selection to produce an optimized dataset ready for forecasting models.

```
+--------------------+
|  Synthetic Data    |  Generates realistic grid demand panel series
+---------+----------+
          |
          ▼
+--------------------+
|  Data Validation   |  Checks ranges, formats, missing values, duplicates
+---------+----------+
          |
          ▼
+--------------------+
|   Data Cleaning    |  Handles outliers, missing data, negative counts
+---------+----------+
          |
          ▼
+--------------------+
| Feature Engineer   |  Temporal, cyclical dates, and safety stocks
+---------+----------+
          |
          ▼
+--------------------+
|  Time-Series Dec   |  Runs statsmodels STL, EMD sifting, DWT wavelets
+---------+----------+
          |
          ▼
+--------------------+
| Feature Selection  |  Strips constant, low variance, & highly correlated features
+---------+----------+
          |
          ▼
+--------------------+
| Optimized Dataset  |  24 high-information predictors saved to processed_dataset.csv
+--------------------+
```

---

## 🗃️ Pipeline Release History

| Version | Release Date | Target Modules | Primary Enhancements |
| :--- | :---: | :---: | :--- |
| **v1.0** | 2026-07-16 | Module 1 | Simulated baseline dataset, outliers capping, chronological data split, safety stock engineering, categorical ordinals. |
| **v1.1** | 2026-07-17 | Module 2 | Grouped time-series sorting, statsmodels STL decomposition, Empirical Mode Decomposition (EMD) IMFs, Discrete Wavelet Transform (DWT), and DWT/EMD static group feature summary generation. |
| **v1.2** | 2026-07-17 | Module 2.5 | Signal reconstruction error validation, feature selector (duplicate, low variance, high correlation choice), dynamic catalog generation, and experiment tracking infrastructure setup. |

---

## 🏗️ Folder Structure Overview

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
│   └── pipeline_version.md   # [This File]
├── src/                      # Source code
│   ├── config/               # Settings loaders
│   ├── data_generation/      # Synthetic generator
│   ├── preprocessing/         # Cleaner, Ordinal encoder, and StandardScaler
│   ├── features/             # Temporal and domain engineers
│   ├── time_series/          # DWT/EMD processors and reconstruction validators
│   ├── feature_selection/    # Redundancy and correlation feature filters
│   ├── evaluation/           # Stable score metrics (MAE, RMSE, MAPE, SMAPE, Pinball)
│   └── models/               # Abstract base model, registry, mock model, training CLI
└── tests/                    # Automated testing suite
```

---

## 📦 Core Generated Artifacts

1. **`artifacts/dwt_transform.joblib`**: Serialized multi-level Discrete Wavelet Transform coefficients.
2. **`artifacts/emd_processor.joblib`**: Serialized Empirical Mode Decomposition IMFs/residual analyzer.
3. **`artifacts/feature_selector.joblib`**: Serialized feature selector listing the chosen 24 active columns.
4. **`data/processed/processed_dataset.csv`**: Enriched, cleaned, and selection-filtered training-ready forecasting inputs.
