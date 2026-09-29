# Comparative Analysis of Transformers and Linear Models for Electricity Consumption Forecasting

## Project Overview

This project implements a comprehensive, reproducible comparative study of time-series forecasting models for smart grid load prediction. It compares four state-of-the-art models across multiple experimental dimensions: horizon length, lookback window, stationarity representation, and temporal shift effects.

### Competing Models

- **DLinear**: Decomposition-based linear model (Cure-Lab LTSF-Linear)
- **NLinear**: Normalization-based linear model (Cure-Lab LTSF-Linear)
- **PatchTST**: Patch-based transformer (yuqinie98/PatchTST)
- **iTransformer**: Inverted transformer with variable-as-token attention (thuml/iTransformer)

### Dataset

- **Source**: Smart grid load simulation (2020–2024, Nigerian smart grid)
- **File**: `DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv`
- **Scope**: 5 years of hourly electricity consumption data with auxiliary features (temperature, rain, holidays, lockdown)
- **Frequency**: Hourly (43,824 samples for 5 years)

## Research Objectives

1. **Horizon Sensitivity**: Analyze accuracy degradation across forecast horizons (96, 192, 336, 720 hours = 4, 8, 14, 30 days)
2. **Lookback Sufficiency**: Identify minimum historical context window for stable predictions
3. **Representation Analysis**: Compare original vs. differenced (stationary) representations and their impact on model performance

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

```bash
# Clone or navigate to project root
cd comparative-forecasting-study

# Install package in development mode (editable)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; import pandas; import sklearn; print('✓ All dependencies installed')"
```

## Quick Start

### 1. Audit Dataset & Preprocessing

```bash
# Run data audit: validate schema, check for gaps, compute statistics
python -m experiments.audit --config configs/study.yaml
```

Expected output:
- Data validation report in `outputs/reports/data_audit.md`
- Audit JSON with checksums and statistics in `outputs/artifacts/`

### 2. Run All Experiments

```bash
# Execute full comparative study (384 runs: 4 models × 4 L × 4 H × 2 branches × 3 seeds)
# Includes resume logic: skips completed runs, retries failures up to 3 times
python -m experiments.run --config configs/study.yaml --experiment all --resume

# Alternative: Run specific experiments
python -m experiments.run --config configs/study.yaml --experiment exp1 --resume
python -m experiments.run --config configs/study.yaml --experiment exp2 --resume
python -m experiments.run --config configs/study.yaml --experiment exp3a --resume
```

Options:
- `--experiment`: one of {all, exp1, exp2, exp3a, exp3b, baseline}
- `--resume`: load prior run manifest; skip completed runs
- `--no-cache`: force retraining (ignore prior completions)

### 3. Generate Reports

```bash
# Compile all findings into comprehensive reports
python -m experiments.report --results outputs/results --output outputs/reports/final_report.md

# This generates:
# - data_audit.md: dataset provenance, quality, schema
# - eda_interpretation.md: exploratory data analysis on training set
# - methodology.md: precise definitions of task, preprocessing, splits, models
# - experiment_1_horizon.md: horizon sensitivity findings
# - experiment_2_lookback.md: lookback sufficiency analysis
# - experiment_3_stationarity_shift.md: representation and shift analysis
# - statistical_analysis.md: Diebold-Mariano statistical tests with Holm correction
# - computational_efficiency.md: training time, inference latency, FLOPs
# - reproducibility.md: run IDs, checksums, environment details
# - final_report.md: executive summary with integrated findings
```

## Project Structure

```
comparative-forecasting-study/
├── configs/                       # Configuration files (YAML)
│   ├── study.yaml                # Master config: paths, split boundaries, L/H ranges, seeds
│   ├── models.yaml               # Model architectures (d_model, n_heads, layers, dropout)
│   └── experiments.yaml           # Experiment definitions
│
├── src/                           # Main source code (Python package)
│   ├── data/                      # Data loading, audit, preprocessing
│   │   ├── loader.py             # CSV loading, timestamp parsing, schema validation
│   │   ├── audit.py              # Data audit statistics
│   │   ├── preprocessor.py       # Scaling, differencing, inversion
│   │   ├── splits.py             # Chronological train/val/test splits
│   │   └── windows.py            # Origin/window generation
│   │
│   ├── models/                    # Model implementations
│   │   ├── base.py               # TimeSeriesAdapter base class
│   │   ├── dlinear.py            # DLinear implementation
│   │   ├── nlinear.py            # NLinear implementation
│   │   ├── patchtst.py           # PatchTST implementation
│   │   ├── itransformer.py       # iTransformer implementation
│   │   └── trainer.py            # Training loop, early stopping, checkpointing
│   │
│   ├── metrics/                   # Evaluation metrics
│   │   ├── accuracy.py           # MAE, MSE, per-lead metrics
│   │   ├── dtw.py                # Dynamic Time Warping (Sakoe-Chiba)
│   │   ├── diebold_mariano.py    # Statistical testing with HAC variance
│   │   └── profiling.py          # Training time, latency, FLOPs, memory
│   │
│   ├── experiments/               # Experiment orchestration
│   │   ├── orchestrator.py       # Run dispatcher, resume, deduplication
│   │   ├── exp1_horizon.py       # Experiment 1: horizon sweep
│   │   ├── exp2_lookback.py      # Experiment 2: lookback sweep
│   │   ├── exp3_repr_shift.py    # Experiment 3A/3B: representation & shift
│   │   └── baseline.py           # Baseline models (persistence, seasonal)
│   │
│   ├── plotting/                  # Visualization
│   │   ├── eda.py                # EDA plots A–E
│   │   ├── results.py            # Experiment result plots
│   │   └── utils.py              # Plot utilities, color schemes
│   │
│   ├── reporting/                 # Report generation
│   │   ├── audit_report.py
│   │   ├── eda_report.py
│   │   ├── methodology_report.py
│   │   ├── exp1_report.py
│   │   ├── exp2_report.py
│   │   ├── exp3_report.py
│   │   └── final_report.py
│   │
│   └── utils/                     # Utilities
│       ├── config.py             # YAML loading, validation
│       ├── device.py             # GPU/CPU detection, seed fixing
│       ├── checksums.py          # MD5 hashing
│       ├── logging.py            # Experiment logging
│       └── env_manifest.py       # Environment capture
│
├── scripts/                       # CLI entry points
│   ├── audit.py                  # Data audit CLI
│   ├── run.py                    # Experiment runner CLI
│   └── report.py                 # Report generator CLI
│
├── tests/                         # Unit tests
│   ├── test_preprocessing.py     # Scaling round-trips, differencing
│   ├── test_splits.py            # Chronological integrity, no leakage
│   ├── test_models.py            # Output shapes, gradients
│   ├── test_metrics.py           # MAE, MSE, DTW, DM tests
│   └── test_baselines.py         # Baseline model validation
│
├── outputs/                       # Experiment outputs (auto-created)
│   ├── runs/                      # Per-run: checkpoints, scalers, curves, configs
│   │   ├── {run_id}/
│   │   ├── run_manifest.json     # Index of all runs, resume logic
│   │   └── ...
│   │
│   ├── predictions/               # CSV: origin, timestamp, lead, target, pred, seed, model
│   │   ├── exp1_predictions.csv
│   │   ├── exp2_predictions.csv
│   │   └── exp3_predictions.csv
│   │
│   ├── results/                   # JSON results files
│   │   ├── exp1_results.json
│   │   ├── exp2_results.json
│   │   └── threshold_table.json
│   │
│   ├── reports/                   # Markdown reports
│   │   ├── data_audit.md
│   │   ├── eda_interpretation.md
│   │   ├── methodology.md
│   │   ├── experiment_1_horizon.md
│   │   ├── experiment_2_lookback.md
│   │   ├── experiment_3_stationarity_shift.md
│   │   ├── statistical_analysis.md
│   │   ├── computational_efficiency.md
│   │   ├── reproducibility.md
│   │   └── final_report.md
│   │
│   ├── plots/                     # Visualization artifacts (PNG, SVG, PDF)
│   │   ├── eda_plots/            # Plots A–E (training data exploration)
│   │   ├── exp1_plots/           # Experiment 1 (horizon sweep)
│   │   ├── exp2_plots/           # Experiment 2 (lookback sweep)
│   │   ├── exp3_plots/           # Experiment 3 (representation & shift)
│   │   └── diagnostics/          # Training curves, sensitivity plots
│   │
│   └── artifacts/                 # Supporting artifacts
│       ├── scalers_branch_o.pkl
│       ├── scalers_branch_d.pkl
│       ├── origin_manifests.json
│       ├── preprocessing_config.json
│       ├── data_checksum.json
│       └── cleaning_audit.json
│
├── DataSet/                       # Raw and processed data
│   ├── CLEAN_smartgrid_5yr_hourly_natural.csv
│   └── splits/
│       ├── train_scaled.csv
│       ├── validation_scaled.csv
│       └── test_scaled.csv
│
├── requirements.txt               # Pinned dependencies
├── setup.py                       # Package installation
├── README.md                      # This file
└── .gitignore
```

## Experimental Design Summary

### Experiment 1: Horizon Sweep (H ∈ {96, 192, 336, 720})
- **Fixed**: L=336, Branch O (original representation)
- **Variables**: Horizon H, Model (4 models), Seed (3 seeds)
- **Research Question**: How does accuracy degrade with longer forecasts?
- **Metrics**: MAE, MSE, per-lead errors, DTW, training cost

### Experiment 2: Lookback Window Sweep (L ∈ {96, 192, 336, 720})
- **Fixed**: H=192, Branch O
- **Variables**: Lookback L, Model, Seed
- **Research Question**: What is the minimum historical context needed?
- **Metrics**: Threshold selection, accuracy vs. computational cost (latency, memory, FLOPs)

### Experiment 3A: Representation Comparison (Branch O vs. Branch D)
- **Fixed**: L=336, H=192
- **Variables**: Representation (original vs. differenced), Model, Seed
- **Research Question**: Does stationarity help? How much information is lost via differencing?
- **Metrics**: Stationarity tests (ADF/KPSS), reconstruction error, per-lead comparison

### Experiment 3B: Temporal Shift Analysis
- **Fixed**: L=336, H=192, Branch O
- **Variables**: Test quarter, Model
- **Research Question**: Are models robust to seasonal/structural shifts in 2024?
- **Analysis**: Correlate prediction errors with measured load shifts per quarter

### Baselines
- **Persistence**: Repeat last observed value
- **Daily Seasonal Naive**: Use value from 24 hours prior
- **Weekly Seasonal Naive**: Use value from 168 hours prior

## Data Governance & Reproducibility

### Train-Test Split
- **Training**: 2020-01-01 to 2023-01-01 (3 years)
- **Validation**: 2023-01-01 to 2024-01-01 (1 year)
- **Test**: 2024-01-01 onwards (held out from any model decisions)

### Preprocessing Rules
- Scaler fitted **only on training data**; applied to val/test without refitting
- Differencing (if used) computed on scaled data; inverse transforms apply unscaling then integration
- All predictions evaluated in original units (kW) for fair comparison

### Seeding & Reproducibility
- **Seeds**: {42, 123, 2024} for variance estimation and reproducibility
- **Run Identification**: SHA-256 hash of config + dataset checksum + preprocessing settings + origin manifest + seed
- **Resume Logic**: Completed runs cached and reused; incomplete runs retried up to 3 times

### Statistical Rigor
- **Diebold-Mariano Tests**: HAC variance with Bartlett kernel, Holm-Bonferroni correction for multiple comparisons
- **Stationarity Tests**: ADF and KPSS on training data (no lookahead)
- **Validity Checks**: Gradient flow, shape contracts, round-trip inversions all verified before full runs

## Technical Notes

### Environment
- **GPU Support**: Models run on GPU if available; CPU fallback automatic
- **Precision**: float32 (standard for time-series forecasting)
- **Batch Size**: 32 (training), deterministic per seed

### Performance Expectations
- **Training Time**: ~5–15 minutes per model/seed (depending on L, H, and hardware)
- **Full Study**: ~384 runs × 5–15 min ≈ 32–96 hours wall time (parallelizable)
- **Inference**: <10ms per batch on modern GPU (batch size 32)

### References

See specification documents for detailed methodology:
- `.kiro/specs/comparative-forecasting-study/requirements.md`
- `.kiro/specs/comparative-forecasting-study/design.md`
- `.kiro/specs/comparative-forecasting-study/tasks.md`

## Contributing

All code must follow these principles:
- No target leakage: future information never used for training/validation decisions
- Deterministic seeding: identical input → identical output
- Full artifact tracking: every run has metadata and can be reproduced
- Honest reporting: failures, limitations, and exploratory exposure all disclosed

## License

[Specify your license, e.g., MIT, GPL-3.0, etc.]

## Contact

For questions or issues, open an issue in the project repository.

---

**Last Updated**: 2024-01-15  
**Project Version**: 0.1.0  
**Status**: Alpha (Task 1.1 initialization complete)
