# POWERGRID Preprocessing Pipeline - Visual Diagram

Below is the conceptual flowchart illustrating the sequential data preparation stages:

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

---

## Detailed Preprocessing Pipeline Sequence

The complete processing workflow, including validation check loops and sequential data splits:

```mermaid
flowchart TD
    %% Define Nodes
    Raw[Raw Dataset: raw_dataset.csv <br>Generated with operational noise]
    ValRaw(Pre-cleaning Validation <br>Identify nulls, duplicates, negatives)
    Clean(Data Cleaning <br>Deduplicate, parse dates, impute medians/modes, IQR outlier caps)
    Temp(Temporal Feature Extraction <br>Year, Month, Cyclical Sin/Cos transforms)
    Feat(Feature Engineering <br>Grouped lags, rolling averages, supply chain metrics)
    ValClean(Post-cleaning Validation <br>Assert zero errors on cleaned stages)
    Split{Chronological Split <br>70% Train / 15% Val / 15% Test}
    Enc(Categorical Encoding <br>Ordinal / One-Hot transforms fit on Train)
    Scale(Numerical Scaling <br>StandardScaler fit on Train)
    TrainOut[Train Dataset: train_dataset.csv]
    ValOut[Val Dataset: val_dataset.csv]
    TestOut[Test Dataset: test_dataset.csv]
    AllOut[Combined Dataset: processed_dataset.csv]

    %% Connect Nodes
    Raw --> ValRaw
    ValRaw --> Clean
    Clean --> Temp
    Temp --> Feat
    Feat --> ValClean
    ValClean --> Split
    
    Split -->|Train Split| Enc
    Split -->|Val Split| Enc
    Split -->|Test Split| Enc
    
    Enc --> Scale
    Scale -->|Train Output| TrainOut
    Scale -->|Val Output| ValOut
    Scale -->|Test Output| TestOut
    
    TrainOut & ValOut & TestOut --> AllOut

    %% Styling
    style Raw fill:#ffb3b3,stroke:#333,stroke-width:2px
    style Split fill:#ffeb99,stroke:#333,stroke-width:2px
    style TrainOut fill:#c2f0c2,stroke:#333,stroke-width:2px
    style ValOut fill:#c2f0c2,stroke:#333,stroke-width:2px
    style TestOut fill:#c2f0c2,stroke:#333,stroke-width:2px
    style AllOut fill:#b3d1ff,stroke:#333,stroke-width:2px
```
