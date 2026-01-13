# ZENN Tornado Damage Analysis

This repository applies the **ZENN (Zentropy Neural Network)** framework to tornado damage prediction on historic buildings, comparing its performance and feature importance with traditional machine learning models (XGBoost, RandomForest).

## Key Results

- **ZENN is statistically equivalent to XGBoost** (Wilcoxon p > 0.05 for all metrics)
- **ZENN identifies different important features**: prioritizes `wall_substrate_u`, `roof_shape_u`, and `retrofit_present_u`
- **ZENN is more robust to hazard removal**: smaller accuracy drop when tornado_EF is excluded

## Repository Structure

```
zennDisasters/
├── scripts/           # Python scripts for analysis
│   ├── zenn_real_model.py       # Core ZENN model adapted for tornado data
│   ├── zenn_cv.py               # 5-Fold Cross-Validation
│   ├── zenn_statistical_test.py # Statistical equivalence testing
│   ├── zenn_feature_importance.py
│   └── ...
├── data/              # Preprocessed input data
│   ├── X_multiscale.csv         # Feature matrix (344 buildings, 34 features)
│   ├── y_consol.csv             # Target: Consolidated damage scale (0-2)
│   └── y_full.csv               # Target: Full damage scale (0-5)
├── outputs/           # Generated results (CSVs, figures)
│   ├── paper_table*.csv         # Tables for paper
│   ├── paper_figure*.png        # Figures for paper
│   ├── zenn_importance_*.csv    # Feature importance rankings
│   └── confusion_matrix_*.png   # Confusion matrix plots
├── poc/               # Proof of concept (synthetic data)
└── original_zenn/     # Original ZENN framework files
```

## Quick Start

### 1. Run Cross-Validation
```bash
cd scripts
python zenn_cv.py y_consol.csv  # Consolidated scale (0-2)
python zenn_cv.py y_full.csv    # Full scale (0-5)
```

### 2. Generate Feature Importance
```bash
python zenn_feature_importance.py
python zenn_hazard_comparison.py  # Hazard-Inclusive vs Hazard-Neutral
```

### 3. Run Statistical Equivalence Test
```bash
python zenn_statistical_test.py
```

### 4. Generate Paper Tables/Figures
```bash
python generate_paper_tables.py
python generate_paper_figures.py
```

## Key Files

| File | Description |
|:---|:---|
| `scripts/zenn_real_model.py` | Core ZENN model with `CoupledModel` class |
| `outputs/zenn_statistical_equivalence.csv` | Statistical equivalence test results |
| `outputs/zenn_importance_hi_aggregated.csv` | Aggregated feature importance (Hazard-Inclusive) |
| `outputs/paper_table7_feature_overlap.csv` | Feature overlap analysis across models |

## Requirements

```
torch
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
scipy
```

## Citation

If you use this code, please cite:
- Original ZENN framework: [WilliamMoriaty/ZENN](https://github.com/WilliamMoriaty/ZENN)
- Tornado vulnerability paper: [Your citation]
