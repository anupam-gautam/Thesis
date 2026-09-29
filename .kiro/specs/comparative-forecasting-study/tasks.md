# Implementation Plan: Comparative Forecasting Study

## Overview

This plan breaks down the comparative time-series forecasting study into 67 discrete, sequenced, and parallelizable tasks across 10 phases. Each task represents 2–4 hours of focused work with clear success criteria, explicit dependencies, and testable outcomes. Tasks are organized to enable parallel execution where possible, with smoke-test and validation checkpoints before full experimental runs. No task fabricates data or skips verification; all findings are evidence-based with honest reporting of any failures or incomplete results.

---

## Phase 1: Project Setup & Data Audit (8 tasks)

- [x] 1.1 Project Initialization
  - Create project structure: `src/`, `configs/`, `tests/`, `outputs/`, `outputs/reports/`, `outputs/plots/`
  - Create `__init__.py` files in all packages
  - Initialize `setup.py` with project metadata (name, version, dependencies: numpy, pandas, scikit-learn, torch, statsmodels)
  - Create `requirements.txt` with pinned versions
  - Create `README.md` with quick-start instructions, CLI reference, and artifact paths
  - _Requirements: Project infrastructure_

- [x] 1.2 Configuration Files
  - Create `configs/study.yaml`: dataset paths, split boundaries (train: 2020–2023-01-01, val: 2023-01-01–2024-01-01, test: 2024-01-01+), lookbacks L ∈ {96, 192, 336, 720}, horizons H ∈ {96, 192, 336, 720}, seeds {42, 123, 2024}, training hyperparameters (batch_size=32, lr=0.001, max_epochs=50, patience=7)
  - Create `configs/models.yaml`: architecture parameters for DLinear, NLinear, PatchTST, iTransformer (d_model, n_heads, n_layers, dropout, kernel sizes)
  - Create `configs/experiments.yaml`: experiment definitions (Exp 1: H sweep, Exp 2: L sweep, Exp 3A: representation branch, Exp 3B: temporal shift quarters)
  - Validate YAML syntax; check all required keys present; report any validation errors
  - _Requirements: Configuration governance_

- [x] 1.3 Utility Modules (Part 1: Config & Device)
  - Implement `src/utils/config.py`: YAML loader, type hints, validation function, config dict with defaults
  - Implement `src/utils/device.py`: GPU/CPU auto-detection, seed-fixing function (set Python, NumPy, PyTorch, DataLoader, device seeds)
  - Implement `src/utils/checksums.py`: MD5 hash function for dataset files, preprocessing settings, and configs
  - Implement `src/utils/logging.py`: experiment status logger (run_id, config hash, start/end time, status)
  - Unit test each utility (synthetic inputs, verify no exceptions)
  - _Requirements: Development infrastructure_

- [x] 1.4 Data Loading & Schema Validation
  - Implement `src/data/loader.py`: read CSV, parse timestamps (infer format), validate schema (check required columns present)
  - Auto-detect timestamp column (datetime-like) and target column (load_kw)
  - Handle timezone information (record if present, default to UTC if not)
  - Test on sample data; verify all required columns (load_kw, temperature, is_rain_day, holidays, lockdown) present or handle gracefully
  - Emit schema summary: columns, dtypes, row count, date range (first/last timestamps)
  - _Requirements: Data integration, schema validation_

- [x] 1.5 Data Audit
  - Implement `src/data/audit.py`: compute audit statistics on supplied CSV
  - Row count, date coverage (first/last timestamps), frequency verification (hourly; check ≥99% of hours present), missing hours detection
  - Duplicate timestamps, missing values per column, numeric ranges (min/max/mean/std per numeric column)
  - If cleaning script has been applied, count affected rows; document cleaning changes
  - Emit audit report (JSON): file path, MD5 checksum, all findings, schema
  - _Requirements: Data quality assessment, provenance_

- [x] 1.6 Chronological Splits
  - Implement `src/data/splits.py`: split data into train (2020–2023-01-01), val (2023-01-01–2024-01-01), test (2024-01-01+)
  - Enforce strict chronological order; reject any reordering
  - Verify no empty splits; raise error if split boundaries fall within data gaps
  - Emit split boundaries, row counts, and date ranges for each split (verify in outputs)
  - _Requirements: Chronological correctness, no target leakage_

- [x] 1.7 Data Audit Report (data_audit.md)
  - Consolidate findings from T1.4, T1.5, T1.6 into `outputs/reports/data_audit.md`
  - Include: file paths, checksums, provenance, schema, units, coverage, frequency, cleaning changes
  - Feature inventory: load_kw (target, units=kW), temperature (auxiliary, units=°C), is_rain_day (binary indicator), holidays (binary), lockdown (binary)
  - Evidence of outages, clipping, anomalies, or unexpected patterns
  - Cross-reference existing plot interpretation if available (`docs/plots/Plot Interpretation.md`)
  - _Requirements: Transparency, traceability_

- [x] 1.8 Verify Cleaning Script for Leakage (smoke test)
  - Inspect `split_and_scale_smartgrid_data.py`: check for full-dataset scaling (should only fit on training), two-sided interpolation, future-informed outlier replacement, backward filling on targets
  - Report findings: "Cleaning script uses correct train-only scaler governance; no detected leakage" or list issues found
  - If leakage detected: create separate causally-correct copy without overwriting supplied file; emit warning
  - Emit `outputs/cleaning_audit.json`: audit outcome, scaler type, leakage status
  - _Requirements: Governance validation, honest reporting_

---

## Phase 2: Preprocessing & Window Generation (6 tasks)

- [x] 2.1 Preprocessor Module (Scaling & Differencing)
  - Implement `src/data/preprocessor.py`: fit StandardScaler on training data only; save/load scaler parameters (pickle)
  - Forward transform: scale training/val/test using training params (no fit on val/test)
  - Branch O (original): return scaled data
  - Branch D (differenced): compute Δx[t] = x[t] - x[t-1], scale differences, save separate scaler for differences
  - Inverse transforms: unscale and reconstruct (Branch D: integrate Δx back to x, then unscale)
  - Round-trip tests: verify `original_data == inverse(forward(original_data))` within ε=1e-6 for both branches
  - _Requirements: Scaler governance, no data leakage, bidirectional transforms_

- [x] 2.2 Window Generation & Origin Manifests
  - Implement `src/data/windows.py`: for each split (train/val/test), generate eligible origins (stride=1)
  - Constraint: origin index + lookback ≤ split boundary; origin + lookback + horizon ≤ end-of-split
  - Generate separate eligible-origin sets for Exp 1, 2, 3A, 3B (accounting for different L/H combinations)
  - Save origin manifests (JSON): timestamps, indices, split assignment, sample counts per split/experiment
  - Verify origin consistency across models: DLinear, NLinear, PatchTST, iTransformer use identical manifests
  - _Requirements: Chronological correctness, no target leakage, consistency_

- [x] 2.3 Unit Tests for Preprocessing
  - Do not implement unit test for now.
   Implement `tests/test_preprocessing.py`:
    - Scaling round-trip: scale then unscale; verify no leakage (unscaled ≈ original)
    - Differencing round-trip: difference, scale, unscale, integrate; verify reconstruction within ε
    - Scaler parameters: verify fit only on training, not on val/test
    - Edge cases: first value (no previous for differencing), final value, single-sample batches
  - Run on small synthetic time series (100 samples) and actual data
  - Report: pass/fail per test, reconstruction errors (mean/max)
  - _Requirements: Quality assurance_

- [x] 2.4 Unit Tests for Splits & Windows
  -Do not implement unit test for now. 
  Implement `tests/test_splits.py`:
    - Chronological correctness: verify train.time < val.time < test.time (no overlap)
    - No target leakage: all training origins have targets within training
    - Validation data: origins may use preceding history from train if lookback extends into train (allowed)
    - Test data: origins may use preceding history from val if lookback extends into val (allowed)
    - Boundary cases: final possible origin per split, first possible origin per split
    - Sample counts: report total eligible origins per experiment, split, L, H combination
  - Run on actual data; report sample counts per experiment
  - _Requirements: Quality assurance_

- [x] 2.5 Preprocessing Artifacts & Checksums
  - Save preprocessing state to `outputs/artifacts/`:
    - Scalers (pickle): one per representation branch (O and D)
    - Origin manifests (JSON): per experiment, split, L/H combination
    - Preprocessing config (JSON): representation type, scaler type, differencing settings
    - Dataset checksum (MD5): verify data consistency across runs
  - Emit summary: "Preprocessing complete; 387 training origins (Exp 1, L=336, H=96), 1024 validation origins, 512 test origins; Branch O and D scalers saved"
  - _Requirements: Reproducibility, governance_

- [x] 2.6 Data Exploration for EDA Context
  - Compute training-data diagnostics (no test data accessed):
    - Rolling mean (7-day, 30-day windows) and variance
    - Autocorrelation at daily (24 lag) and weekly (168 lag) offsets
    - Seasonal profiles: mean load by hour (0–23), weekday vs. weekend differences
  - Generate exploratory plots (A–E): entire series + rolling mean, deterministic week sample, calendar month sample, daily average profiles, hourly average + std, weekday×hour heatmap
  - Emit plots as PNG; compute numerical evidence (peak hour, weekday/weekend delta %, seasonal variation coefficient)
  - Report: "Training data shows daily periodicity with peak at 18±2 hours; weekday/weekend diff ~8%; seasonal coefficient 0.23"
  - _Requirements: Context for interpretation, exploratory disclosure_

---

## Phase 3: Model Implementations & Trainer (8 tasks)

- [ ] 3.1 Base Model Adapter Class
  - Implement `src/models/base.py`: abstract `TimeSeriesAdapter` class
  - Define interface: `forward(x)` where x ∈ [batch, lookback, channels], output ∈ [batch, horizon, channels]
  - Methods: `get_flops()`, `get_parameter_count()` (return ints or None if not computed)
  - All concrete models inherit from this class; enforce shape contract
  - _Requirements: Architecture standardization_

- [ ] 3.2 DLinear & NLinear Implementation
  - Implement `src/models/dlinear.py`: decompose into trend (linear regression) + seasonal (moving average, kernel=25), individual_channel_weights=False
  - Implement `src/models/nlinear.py`: remove last value, apply linear layer, add back, individual_channel_weights=False
  - Both inherit from `TimeSeriesAdapter`; verify output shapes [batch, L, C] → [batch, H, C]
  - Test gradient flow on dummy input; verify no NaN/Inf
  - _Requirements: Model implementations, gradient validation_

- [ ] 3.3 PatchTST Implementation
  - Implement `src/models/patchtst.py`: patch embedding (patch_length=16, stride=8), multi-head attention (d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1)
  - Channel-independent processing; flatten then reshape per patch
  - Inherit from `TimeSeriesAdapter`; verify output shapes and gradient flow
  - _Requirements: Transformer baseline_

- [ ] 3.4 iTransformer Implementation
  - Implement `src/models/itransformer.py`: inverted attention (variables as tokens, not time steps), cross-variable attention, 2 layers
  - Key params: d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1
  - Reshape input [batch, L, C] → [batch, C, L] for attention over time; reverse reshape for output
  - Inherit from `TimeSeriesAdapter`; verify output shapes and gradient flow
  - _Requirements: Transformer baseline with variable-as-token inversion_

- [ ] 3.5 Training & Checkpoint Logic
  - Implement `src/models/trainer.py`: training loop with Adam optimizer (lr=0.001), batch_size=32, max_epochs=50, early stopping (patience=7)
  - Validation every epoch on validation set; checkpoint best validation MSE model
  - Save: model checkpoint (PyTorch), optimizer state, training curves (CSV: epoch, train_loss, val_loss), metadata (final_epochs, best_loss, wall_time_sec)
  - Seed fixing: before training, call device.py seed setter
  - Precision: float32 throughout
  - _Requirements: Reproducible training, checkpoint governance_

- [ ] 3.6 Unit Tests for Models
  - Do not implement unit test for now.
  Implement `tests/test_models.py`:
    - Output shapes: input [32, L, C] → output [32, H, C] for each of 4 models with L ∈ {96, 336}, H ∈ {96, 192}
    - Gradient flow: backward pass succeeds, no NaN/Inf gradients on each model
    - Tiny-signal learning: each model fits synthetic constant + noise signal (100 samples) in 1 epoch
    - Parameter counts: report total and trainable params per model
  - Run on CPU; report pass/fail per test
  - _Requirements: Quality assurance_

- [ ] 3.7 Unit Tests for Trainer
  - Do not implement unit test for now.
  Implement `tests/test_trainer.py`:
    - Training loop completeness: training without error (2 epochs, tiny dataset)
    - Checkpointing: model state saved and restored correctly
    - Early stopping: triggers after patience exceeded (test on non-learning signal)
    - Training curves CSV: valid CSV with epochs, train_loss, val_loss columns
    - Seed reproducibility: two runs with identical seed produce identical results
  - Run on CPU; report pass/fail per test
  - _Requirements: Quality assurance_

- [ ] 3.8 Model Validation & Fidelity Smoke Test
  - Run smoke test: train each model (DLinear, NLinear, PatchTST, iTransformer) for 2 epochs on tiny dataset (100 training samples, L=96, H=96, 1 feature)
  - Verify: output shapes, gradient flow, loss decreasing (convergence direction)
  - Report: "All 4 models passed smoke test; architecture and gradient flow verified. DLinear: loss 0.45→0.41, NLinear: 0.48→0.44, PatchTST: 0.52→0.48, iTransformer: 0.50→0.46"
  - _Requirements: Architecture validation before full runs_

---

## Phase 4: Metrics & Statistical Testing (6 tasks)

- [ ] 4.1 Accuracy Metrics (MAE, MSE, Per-Lead)
  - Implement `src/metrics/accuracy.py`:
    - MAE: mean(|predicted - observed|), units = kW
    - MSE: mean((predicted - observed)²), units = kW²
    - Per-lead: separate MAE/MSE for each forecast lead k ∈ {1, 2, ..., H}
    - Aggregation: per-seed mean over all origins, then across 3 seeds (mean ± std)
  - Return dictionary: {metric_name: {seed_id: value, mean: float, std: float}}
  - _Requirements: Accuracy evaluation_

- [ ] 4.2 DTW Implementation (Sakoe–Chiba Constrained)
  - Implement `src/metrics/dtw.py`:
    - Sakoe–Chiba band: ±24 hours (±24 time steps at hourly resolution)
    - Local cost: absolute error |predicted[k] - observed[k]|
    - Steps allowed: (1,0), (0,1), (1,1)
    - Endpoint constraints: (0,0) → (H-1, H-1) (both sequences must be fully consumed)
    - Path normalization: cumulative_cost / path_length
    - Deterministic tie-breaking when multiple paths have equal cost (preference: diagonal > left > down)
    - Sampling: every 24 hours for large H; report actual sample count
  - _Requirements: Temporal distortion penalty_

- [ ] 4.3 Unit Tests for DTW
  - Do not implement unit test for now.
   Implement `tests/test_metrics.py` (DTW section):
    - Self-distance: DTW(x, x) = 0 for synthetic and real data
    - Sakoe–Chiba constraint: verify no paths outside band; report actual band enforcement
    - Deterministic output: same forecast/observed pair → identical DTW value (across runs)
    - Edge cases: H=1, H=96, H=720
  - Run on synthetic sequences (constant, linear trend, sine waves)
  - _Requirements: Quality assurance_

- [ ] 4.4 Diebold–Mariano Statistical Test
  - Implement `src/metrics/diebold_mariano.py`:
    - Loss differential: d[t] = loss_A[t] - loss_B[t] (MSE per origin; must be same set of origins for both models)
    - HAC variance: Bartlett kernel, bandwidth ≥ H-1 (Newey-West formula; use statsmodels if available)
    - DM statistic: mean(d) / sqrt(Var_HAC(d) / n)
    - p-value: two-tailed t-test, n-2 degrees of freedom (from t-distribution)
    - Holm correction: sort p-values, apply threshold α/(k+1-i) where i is rank
    - Outputs: DM statistic, p-value, adjusted p-value (Holm), bandwidth, sample size, validity flag (True if variance > 0)
  - _Requirements: Statistical comparison, multiple-comparison control_

- [ ] 4.5 Unit Tests for Statistical Testing
  - Do not implement unit test for now. 
  Implement `tests/test_metrics.py` (DM section):
    - Hand-calculated DM case: synthetic d[t] series → verify statistic, p-value against manual calculation
    - Degenerate variance: when Var ≈ 0, return validity=False; do not return p=0
    - Holm correction: verify threshold calculation for k=6 pairs
    - Seed sensitivity: run DM on same data with multiple seeds; results should be identical
  - Run on real experiment outputs
  - _Requirements: Quality assurance_

- [ ] 4.6 Computational Profiling Metrics
  - Implement `src/metrics/profiling.py`:
    - Training time: wall time (seconds), actual epochs trained
    - Inference latency: batch size 1 and 32, no-grad mode, ≥20 warm-ups, ≥100 timed repetitions, report median and p95 (ms per window)
    - Parameter count: total and trainable (integers)
    - FLOPs: forward-pass estimation (use PyTorch profiler or manual estimation); report tool/version and whether partial
    - Memory: peak device training memory (GB), peak device inference memory (GB)
    - Hardware metadata: device name, OS, PyTorch version, number of threads
  - Return dictionary with all metrics
  - _Requirements: Computational cost analysis_

---

## Phase 5: Experiment Orchestration (5 tasks)

- [ ] 5.1 Run Orchestrator & Resume Logic
  - Implement `src/experiments/orchestrator.py`:
    - Expand config → individual runs: (model, L, H, representation, seed)
    - Compute run_id: SHA-256 hash of (config, dataset_checksum, preprocessing_checksum, origins_checksum, seed)
    - Load `run_manifest.json` if exists; check for prior completed runs
    - Deduplication: skip completed runs; load predictions from cache
    - Queue pending runs; execute serially or parallel (up to 5 concurrent with multiprocessing)
    - Resume: restart only incomplete runs (up to 3 retries per run on failure)
    - Save updated `run_manifest.json` at end: {run_id: {config, status, predictions_path, error (if failed)}}
  - _Requirements: Reproducibility, fault tolerance, efficiency_

- [ ] 5.2 Experiment 1 (Horizon Sweep)
  - Implement `src/experiments/exp1_horizon.py`:
    - H ∈ {96, 192, 336, 720}, L=336, original representation (Branch O)
    - Generate origins per H (constraint: origin + L + H ≤ split boundary)
    - Train all 4 models × 4 horizons × 3 seeds = 48 runs
    - For each run: compute MAE, MSE, DTW on test set; per-lead MAE/MSE
    - Aggregate across seeds: mean ± std
    - Save: `outputs/results/exp1_results.json` (structured results), `outputs/results/exp1_predictions.csv` (all predictions)
    - Report: completion status, sample counts, aggregated metrics
  - _Requirements: Horizon sensitivity analysis_

- [ ] 5.3 Experiment 2 (Lookback Sweep)
  - Implement `src/experiments/exp2_lookback.py`:
    - L ∈ {96, 192, 336, 720}, H=192, original representation (Branch O)
    - Generate origins per L
    - Train all 4 models × 4 lookbacks × 3 seeds = 48 runs
    - For each run: compute MAE, MSE, training time, inference latency, parameter count, FLOPs
    - Threshold selection: for each model, find L_min where val_MSE ≤ best_val_MSE × 1.05
    - Save: `outputs/results/exp2_results.json`, `outputs/results/exp2_predictions.csv`, `outputs/results/threshold_table.json`
    - Report: threshold findings, cost scaling (latency vs. L, params vs. L)
  - _Requirements: Lookback sufficiency analysis, computational cost analysis_

- [ ] 5.4 Experiment 3A & 3B (Representation & Temporal Shift)
  - Implement `src/experiments/exp3_repr_shift.py`:
    - **3A**: Branch O vs. Branch D; L=336, H=192; train models on both branches; 4 models × 2 branches × 3 seeds = 24 runs
      - Perform ADF/KPSS tests on training data (Branch O vs. D); report p-values, critical values, interpretation
      - Compute MAE/MSE on test set for each branch/model/seed
      - Reconstruction analysis: compute error accumulation when integrating differences
    - **3B**: Predefine test quarters (Q1, Q2, Q3, Q4 of 2024) before viewing errors; no additional training
      - Evaluate original-branch test predictions on per-quarter basis
      - Compute shift magnitude per quarter: mean_load, std_load, peak_hour, weekday_delta
      - Compute error metrics (MAE/MSE) per quarter
      - Correlate error increase with measured shift
    - Save: `outputs/results/exp3a_results.json`, `outputs/results/exp3b_shift_analysis.json`
    - Report: stationarity findings, representation benefits, shift correlations
  - _Requirements: Representation benefit analysis, robustness to temporal shift_

- [ ] 5.5 Baseline Models
  - Implement `src/experiments/baseline.py`:
    - **Persistence**: predict last observed value for all leads (ŷ[t+k] = y[t] for all k)
    - **Daily seasonal naive**: use corresponding hour from previous day (ŷ[t+k] = y[t+k-24])
    - **Weekly seasonal naive**: use corresponding hour from previous week (ŷ[t+k] = y[t+k-168])
    - Evaluate on identical origin manifests as research models (Exp 1 origins)
    - Verify: baselines do not read future values; all predictions within valid time range
    - Compute MAE/MSE for each baseline on test set
    - Save: `outputs/results/baseline_predictions.csv`, `outputs/results/baseline_metrics.json`
    - Report: baseline performance (kW MAE), model comparison anchor
  - _Requirements: Baseline establishment for model comparison_

---

## Phase 6: Plotting & Visualization (4 tasks)

- [ ] 6.1 EDA Plots (Plots A–E)
  - Implement `src/plotting/eda.py`:
    - **Plot A**: Entire training series + 30-day rolling mean, labeled axes, legend, datetime x-axis
    - **Plot B**: Deterministically selected complete week (e.g., first full week of training)
    - **Plot C**: Deterministically selected complete calendar month (e.g., January of training year)
    - **Plot D1**: Daily average load over time, document incomplete-day treatment
    - **Plot D2**: Average load by hour (0–23) with std as confidence bands
    - **Plot E**: Weekday × hour heatmap (mean load per cell; mark weekends differently)
    - All plots: training data only, no test data
    - Export PNG (300 DPI), SVG, PDF; compute numerical evidence (peak hour, ranges, seasonal coefficients)
    - Save to `outputs/plots/`
  - _Requirements: Visual data exploration, training-set-only discipline_

- [ ] 6.2 Result Plots (Experiments)
  - Implement `src/plotting/results.py`:
    - **Exp 1**: Accuracy (MAE/MSE) vs. horizon H (4-panel per metric), per-lead error curves (one curve per model), model comparison
    - **Exp 2**: Accuracy (MAE/MSE) vs. lookback L, latency vs. L, parameter count vs. L, threshold annotations
    - **Exp 3A**: Accuracy comparison (Branch O vs. D), per-model breakdown, box plots or bar charts
    - **Exp 3B**: Error vs. quarter, shift magnitude vs. quarter, correlation plot
    - **Accuracy vs. cost Pareto**: 2D plots (MSE on y-axis, training time / inference latency / parameter count on x-axis); identify efficient models
    - Color scheme: DLinear=blue, NLinear=orange, PatchTST=green, iTransformer=red
    - Save PNG (300 DPI), SVG, PDF to `outputs/plots/`
  - _Requirements: Result visualization, model comparison_

- [ ] 6.3 Plot Data Export
  - Implement `src/plotting/utils.py`:
    - For each plot generated: export underlying (X, Y) values as CSV
    - Enable reproducibility and downstream analysis
    - Store alongside PNG/SVG/PDF with matching filenames
    - CSV format: header row + data rows (numeric values, no graphics)
  - _Requirements: Reproducibility, data transparency_

- [ ] 6.4 Diagnostic Plots (Training Curves, Hyperparameter Sensitivity)
  - Implement `src/plotting/diagnostics.py`:
    - **Training curves**: epoch vs. train/val loss per seed, per model (separate subplot per model)
    - **Hyperparameter sensitivity** (optional, labeled separately): accuracy vs. L (optional grid search), not part of primary results
    - Save all as PNG (300 DPI), SVG
    - Collect in `outputs/plots/diagnostics/`
  - _Requirements: Training monitoring, optional sensitivity analysis_

---

## Phase 7: Report Generation (10 tasks)

- [ ] 7.1 Data Audit Report (data_audit.md)
  - Generate `outputs/reports/data_audit.md` (already started in T1.7):
    - Input paths, file sizes, checksums (MD5)
    - Provenance: source, collection method, time period (2020–2024)
    - Schema: columns, dtypes, row count, date range, time zone, frequency (hourly)
    - Data quality: duplicates, missing hours, missing values per column, numeric ranges
    - Cleaning script findings: changes made (if any), affected row counts, unresolved issues
    - Feature inventory:
      - load_kw (target, units=kW, always available at t for forecast at t+1 onward)
      - temperature (auxiliary, units=°C, available at forecast time)
      - is_rain_day (binary indicator, 0 or 1)
      - holidays (binary indicator, 0 or 1)
      - lockdown (binary indicator, 0 or 1)
    - Evidence of outages, clipping, impossible values, anomalies
  - _Requirements: Transparency, traceability, documentation_

- [ ] 7.2 EDA Interpretation Report (eda_interpretation.md)
  - Generate `outputs/reports/eda_interpretation.md`:
    - Plots A–E with captions and numerical evidence
    - Training-data diagnostics: rolling mean/var, ACF at 24 and 168 lags, seasonal profiles
    - Measured evidence: peak hour (±2 hours), weekday/weekend delta (%), seasonal variation (coefficient of variation)
    - Relevance to forecasting: which patterns are predictable?
    - Interpretation limits: "Exploration limited to training data; test set not inspected"
    - Verify prior exploratory claims: daily/weekly structure confirmed?, peaks present?, weekday–weekend differences present?
    - **Disclose exploratory exposure**: "Training data was inspected during project setup (EDA); this exploratory analysis may bias interpretation of test results"
  - _Requirements: Transparency, acknowledgment of exploration, limitation documentation_

- [ ] 7.3 Methodology Report (methodology.md)
  - Generate `outputs/reports/methodology.md`:
    - Task definition: forecast load_kw (target) using historical load and auxiliary features (temperature)
    - Target definition: load_kw ∈ [0, ∞) kW, real-valued; availability at prediction time (only past/concurrent observed values)
    - Preprocessing: StandardScaler (fit on training only), differencing (Branch D), inverse transforms
    - Scaler governance: fit on training, applied to val/test without refitting
    - Split definition: train (2020–2023-01-01), val (2023-01-01–2024-01-01), test (2024-01-01+); row counts and coverage
    - Window definition: origin as last observed timestamp; input window [t-L+1, t], output window [t+1, t+H]; stride=1
    - Eligible origin constraints per experiment: origin + L ≤ split boundary; origin + L + H ≤ end-of-split
    - Baselines: persistence (repeat last), daily seasonal naive (24 lags), weekly seasonal naive (168 lags)
    - Models: DLinear (decomposition), NLinear (remove-add back), PatchTST (patch embedding + attention), iTransformer (inverted attention)
      - Source repos, canonical implementations, architecture hyperparameters, training config (batch_size=32, lr=0.001, max_epochs=50, patience=7)
    - Validation checks: chronological correctness, scaler governance, origin consistency, model output shapes, gradient flow, tiny-signal learning
    - Seed strategy: 3 seeds {42, 123, 2024} for reproducibility and variance estimation
  - _Requirements: Clarity, reproducibility, governance documentation_

- [ ] 7.4 Experiment 1 Report (experiment_1_horizon.md)
  - Generate `outputs/reports/experiment_1_horizon.md`:
    - Research question: "How do accuracy and computational cost vary with forecast horizon H?"
    - Experimental setup: H ∈ {96, 192, 336, 720}, L=336, original representation
    - Origin and sample counts: report per split, per H value
    - Results table: H | model | seed | MAE (kW) | MSE (kW²) | DTW | training_time_sec | inference_latency_ms
    - Aggregated results across seeds: mean ± std
    - Plots: MAE vs. H (4 models), MSE vs. H, per-lead error curves (H=720), accuracy vs. training cost (Pareto)
    - Per-lead analysis: identify horizon-dependent model strengths (e.g., "DLinear strongest at H=96; NLinear degrades after H=192")
    - Interpretation: "Which models maintain accuracy at long horizons? Which degrade rapidly?"
    - Statistical comparison: DM tests for final lead (H=720, lead k=H) comparing all 6 pairs, Holm-corrected p-values
    - Limitations: single dataset, exploratory exposure (training EDA completed), limited seed budget (3 seeds)
    - Direct answer to research question: "Accuracy degrades with horizon; NLinear best at H≤192; iTransformer best at H=720 but with X% higher cost"
  - _Requirements: Clarity, evidence-based conclusion, honest limitations_

- [ ] 7.5 Experiment 2 Report (experiment_2_lookback.md)
  - Generate `outputs/reports/experiment_2_lookback.md`:
    - Research question: "How much observed history is useful? What is the practical minimum lookback?"
    - Experimental setup: L ∈ {96, 192, 336, 720}, H=192, original representation
    - Results table: L | model | seed | MAE (kW) | MSE (kW²) | training_time_sec | latency_ms | params | FLOPs | memory_GB
    - Threshold table: model | L_threshold | val_MSE_best | val_MSE_at_threshold | efficiency_notes
    - Plots: accuracy vs. L (per model), latency vs. L, parameter count vs. L, accuracy-cost Pareto (MSE vs. training time)
    - Practical threshold interpretation: "DLinear reaches 95% best accuracy with L=192 (2 weeks); additional history provides marginal gains at increasing cost"
    - Cost analysis: training time scaling, inference latency scaling, memory usage vs. L
    - Efficiency findings: "NLinear cheapest; PatchTST requires GPU memory beyond available budget"
    - Direct answer: "Practical minimum history recommendation: L=192 for DLinear/NLinear, L=336 for Transformers (if GPU available)"
  - _Requirements: Practical utility, cost-benefit analysis, accessibility assessment_

- [ ] 7.6 Experiment 3A & 3B Report (experiment_3_stationarity_shift.md)
  - Generate `outputs/reports/experiment_3_stationarity_shift.md`:
    - **3A: Representation & Stationarity**
      - Research question: "Does differencing representation help forecasting accuracy? Is stationarity a factor?"
      - ADF/KPSS results on training data (Branch O and D): test statistics, p-values, critical values, interpretation (stationary?, nonstationary?)
      - Results table: representation | model | seed | MAE (kW) | MSE (kW²) | reconstruction_error
      - Accuracy comparison: Branch O vs. Branch D, per model; highlight winners/losers
      - Reconstruction analysis: integrating differences accumulates error; report typical reconstruction error
      - Interpretation: "Differencing benefits NLinear (X% improvement) but marginal for others; stationarity tests show differencing reduces but does not eliminate nonstationarity"
      - Caveat: "Loss-space optimization differences and reconstruction error prevent direct attribution of accuracy gains to stationarity"
    - **3B: Temporal Distribution Shift**
      - Research question: "Does model performance vary with measured changes in load distribution across test quarters?"
      - Test quarters: Q1, Q2, Q3, Q4 of 2024 (predefined before error inspection)
      - Shift table: quarter | mean_load (kW) | std_load (kW) | peak_hour | peak_load (kW) | weekday_delta (%) | notes
      - Error table: quarter | model | MAE (kW) | MSE (kW²)
      - Correlation analysis: plot error vs. measured shift magnitude; compute correlation coefficient
      - Period comparison: do recurring patterns (Q1 2024 vs. Q1 training) show stable error?
      - Interpretation: "Error increases in Q2 where load mean increases by 8%; likely task difficulty rather than model drift sensitivity"
      - Caveat: "No strong evidence of model-specific robustness to shift in this 1-year test window; longer evaluation needed"
    - _Requirements: Stationarity assessment, temporal stability analysis, honest uncertainty_

- [ ] 7.7 Statistical Analysis Report (statistical_analysis.md)
  - Generate `outputs/reports/statistical_analysis.md`:
    - DM test results: all 6 pairwise comparisons per experiment condition
    - Table: comparison (A vs. B) | mean_diff (kW²) | DM_stat | p_value | adj_p_value (Holm) | bandwidth | n_samples | validity
    - Holm correction: show threshold α/(k+1-i) for k=6 pairs; identify which comparisons survive correction
    - Seed sensitivity: DM ranges (min–max) across 3 seeds {42, 123, 2024}
    - Effect size: mean difference in original units (kW²); practical significance vs. baseline variance
    - Multiple comparison control: explain Holm-Bonferroni correction and family-wise error rate
    - Validity assessment: HAC bandwidth convergence, sample size adequacy, temporal dependence handling
    - Caution statements: "Overlapping forecast origins create temporal dependence; DM tests assume HAC variance estimation captures this; verify bandwidth ≥ H-1"
    - Interpretation guidance: "Surviving comparisons indicate statistically significant differences; non-surviving comparisons inconclusive"
  - _Requirements: Statistical rigor, transparency, honest caveats_

- [ ] 7.8 Computational Efficiency Report (computational_efficiency.md)
  - Generate `outputs/reports/computational_efficiency.md`:
    - Hardware spec: device (GPU model or CPU), OS, total RAM/VRAM (GB), PyTorch version, thread count, precision (float32)
    - Training time: wall time (mean ± std across seeds), actual epochs, median epoch time (sec/epoch)
    - Inference latency: batch 1 and 32, median and p95 (ms/window), throughput (windows/sec)
    - Parameter count: per model and per (L, H) configuration; table format
    - FLOPs: per model, forward-pass at batch=1; profiler tool and version; note if partial estimates
    - Memory: peak device training (GB), peak device inference (GB); report out-of-memory (OOM) issues if encountered
    - Cost scaling: training time vs. L (table), inference latency vs. L (plot), parameters vs. L
    - Accuracy–cost Pareto: 2D plots (MSE on y-axis, training time / inference latency / parameter count on x-axis per separate plot); identify efficient models/configs
    - Hardware limitations: "iTransformer requires GPU memory exceeding available budget for L=720; run on CPU feasible but slow (estimated X hours)"
    - Interpretation: "DLinear achieves NLinear-level accuracy at 3x lower training cost; PatchTST middle ground; iTransformer not justified unless extremely high accuracy required"
  - _Requirements: Operational guidance, hardware accessibility, cost transparency_

- [ ] 7.9 Reproducibility Report (reproducibility.md)
  - Generate `outputs/reports/reproducibility.md`:
    - Execution status: "Smoke test complete; full matrix pending" or "All runs completed successfully" with date/time
    - CLI commands to reproduce:
      ```bash
      python -m src.experiments.orchestrator --config configs/study.yaml --phase audit
      python -m src.experiments.orchestrator --config configs/study.yaml --phase run --experiment all --resume
      python -m src.plotting.results --results outputs/results --output outputs/plots/
      python -m src.reports.generate --results outputs/results --artifacts outputs/artifacts --output outputs/reports/
      ```
    - Environment: Python version, PyTorch version, scikit-learn version, numpy version, pandas version; run `pip freeze` output saved to `environment.txt`
    - Artifact index: paths to checkpoints (per run), scalers (Branch O/D), configs (study/models/experiments), training curves CSV, predictions CSV, plots PNG/SVG
    - Run status: total runs queued, completed, skipped (deduplicated), failed, pending
    - Failed runs (if any): exception messages, recommended debugging steps (e.g., "Run 42 OOM on inference latency; reduce batch size in trainer.py and retry")
    - Resume command: exact command to restart incomplete runs (same as run, with `--resume`)
    - References: citations for LTSF-Linear (Zeng et al.), PatchTST (Nie et al.), iTransformer (Liu et al.), Diebold–Mariano (Diebold & Mariano 2002), Holm correction (Holm 1979)
    - Data availability: path to input CSV (`DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv`), note that it is simulated data (REFERENCE: Charushila et al. or generator), link to public dataset if available
  - _Requirements: Transparency, runability, citation, data provenance_

- [ ] 7.10 Final Report (final_report.md)
  - Generate `outputs/reports/final_report.md` (self-contained, top-level orchestration):
    - **Executive Summary** (2–3 paragraphs): overview of findings, key comparisons between models, main conclusion (e.g., "DLinear matches Transformer accuracy at fraction of cost; Transformers not justified for this dataset")
    - **Objectives & Hypotheses** (3 research objectives stated as testable hypotheses):
      1. "H1: Transformer models (PatchTST, iTransformer) achieve significantly lower test MSE than linear models (DLinear, NLinear)"
      2. "H2: Differencing representation improves forecasting accuracy more for linear models than Transformers"
      3. "H3: Model performance is robust to temporal distribution shifts in test data"
    - **Dataset & Limitations**: provenance (simulated via generator or reference), simulation caveat, verified audit findings, exploratory exposure disclosure (EDA on training), single-site limitation
    - **Task & Preprocessing**: target (load_kw, units=kW), channels (load history, temperature), chronological splits, scaler governance (fit on training only), no data leakage verified
    - **Models & Configuration**: table with model names, source repos, canonical implementations, architecture hyperparameters (d_model, n_heads, n_layers, dropout), shared training config (batch_size=32, lr=0.001)
    - **Experiments** (summary of each):
      - Exp 1 (Horizon): summary findings (e.g., "Accuracy degrades uniformly with horizon; X% MAE increase from H=96 to H=720")
      - Exp 2 (Lookback): practical L threshold, efficiency trade-off (e.g., "L=192 sufficient for 95% accuracy; L=336 provides marginal 2% improvement")
      - Exp 3A (Representation): stationarity findings, accuracy comparison (e.g., "Differencing benefits NLinear but not iTransformer; ADF tests show partial stationarity improvement")
      - Exp 3B (Temporal Shift): measured shift analysis, error correlation (e.g., "No evidence of systematic degradation; error variation within baseline noise")
    - **Metrics & Statistical Testing**: define MAE/MSE/DTW, describe DM procedure, Holm correction, validity checks (HAC bandwidth, sample size)
    - **Accuracy vs. Cost Pareto** (plot and table): compare models on accuracy (test MSE) and computational cost (training time, inference latency, parameters); identify efficient frontiers
    - **Evidence on Transformer Justification** (primary research question, direct answer):
      - Summarize test MSE gains (if any) for PatchTST and iTransformer vs. DLinear/NLinear (e.g., "PatchTST: +2% accuracy, 10x training cost; iTransformer: +1% accuracy, 50x training cost, GPU required")
      - Compare training time, inference latency, parameter counts
      - **Direct recommendation**: "For this dataset, DLinear matches PatchTST accuracy at 10x lower training cost and 3x lower inference latency; Transformer overhead not justified"
    - **Threats to Validity** (itemized):
      - Simulation: single simulated dataset; cannot generalize to actual grids
      - Single dataset: no multi-site analysis; dataset-specific phenomena may dominate findings
      - Exploratory exposure: training data inspected during setup (EDA); test set not completely unseen; may bias interpretation
      - Limited seed budget: 3 seeds only; high-variance estimators may have wide confidence intervals
      - Overlapping forecasts: temporal dependence in evaluation; DM tests assume HAC variance handles this (no guarantee)
      - Loss vs. representation: attributed improvements may reflect loss-space optimization differences, not representation benefits
      - Residual nonstationarity: differencing does not guarantee strict stationarity; some nonstationarity remains
      - Channel treatment: temperature as exogenous only; cross-variable interaction (multivariate) not fully evaluated
      - Hardware dependence: results specific to GPU/device used; inference latency not portable to different hardware
      - Hyperparameter tuning: training config fixed (not tuned per model); models may benefit from individual hyperparameter optimization
    - **Conclusions** (address each objective directly):
      - Objective 1 (Transformers advantage): "Hypothesis NOT supported; test MSE shows DLinear and NLinear on par with or better than Transformers"
      - Objective 2 (Representation benefit): "Hypothesis partially supported; differencing helps NLinear but not iTransformer"
      - Objective 3 (Robustness to shift): "Hypothesis not testable; no systematic shift detected in test data"
      - **Practical finding**: "Recommended model: DLinear with L=192, H=192; achieves competitive accuracy with minimal computational cost"
      - **Negative finding reported explicitly**: "Transformer models not justified for operational deployment on this dataset; computational cost exceeds accuracy gains"
    - **Reproducibility** (operational):
      - Exact CLI commands to reproduce all runs and reports
      - Environment specs and `environment.txt` location
      - Data access: path to CSV, link if publicly available, note on simulation/provenance
      - Code availability: repository (if public) or local path for reproducibility
    - **Artifact Index**: list all saved files (checkpoints, scalers, configs, predictions, plots, reports); checksums (MD5) for dataset, configs
    - **References**: LTSF-Linear (Zeng et al. 2023), PatchTST (Nie et al. 2023), iTransformer (Liu et al. 2023), Diebold–Mariano (Diebold & Mariano 2002), Holm correction (Holm 1979), ADF/KPSS (statsmodels), DTW (Sakoe & Chiba 1978)
    - **Missing or Failed Runs** (if any): report exactly which runs (model, L, H, seed) incomplete and why; provide resume command; do NOT claim completion if incomplete
    - **Success**: "Study complete; all smoke tests passed; X of Y experiment runs completed successfully; all reports generated and archived"
  - _Requirements: Comprehensive, evidence-based, honest assessment of findings and limitations_

---

## Phase 8: Validation & Testing (4 tasks)

- [ ] 8.1 End-to-End Smoke Test
  - Execute all modules in sequence on tiny dataset (100 samples, 1 feature):
    - Load CSV → audit → split (chronological) → preprocess (both branches) → window generation
    - Train all 4 models for 2 epochs on 100 training samples, L=96, H=96
    - Compute metrics (MAE, MSE, DTW, training time, inference latency)
    - Verify no errors; report: "Smoke test passed; all 4 models complete successfully. Data flow verified; no errors."
  - If any error: report exact error message, step where failed, and recommended fix
  - _Requirements: Architectural validation, error detection_

- [ ] 8.2 Validation Checklist (10 items)
  - Execute checklist from design document:
    1. ✓ Chronological correctness: split boundaries, no target leakage verified
    2. ✓ Scaler governance: fit only on training; val/test use training scaler without refitting
    3. ✓ Differencing round-trip: forward and backward transform within ε=1e-6
    4. ✓ Origin consistency: identical manifests across models (DLinear/NLinear/PatchTST/iTransformer)
    5. ✓ Model output shapes: [batch, L, channels] → [batch, H, channels] for all models
    6. ✓ Gradients: finite, no NaN/Inf detected during backward pass
    7. ✓ Tiny-signal learning: each model fits synthetic constant + noise signal (100 samples) in ≤2 epochs
    8. ✓ Seasonal baselines: persistence and seasonal-naive models do not read future values
    9. ✓ DTW: self-distance = 0 for test sequences
    10. ✓ Resume logic: deduplication avoids re-training completed runs; run_manifest.json accurate
  - Report status: ✓ (pass) or ✗ (fail) per item with explanation
  - _Requirements: Quality assurance, confidence in orchestration_

- [ ] 8.3 Data Integrity Tests
  - Run all unit tests: `tests/test_preprocessing.py`, `tests/test_splits.py`
  - Report: pass/fail per test; coverage percentage
  - If any failures: stop and report exact test name, assertion, and data sample (e.g., "test_scaling_round_trip FAILED: unscaled[0] = -0.5, expected ≤ 1e-6 absolute error, original[0] = 0.0")
  - _Requirements: Data governance validation_

- [ ] 8.4 Model Correctness & Learning Tests
  - Run all unit tests: `tests/test_models.py`, `tests/test_trainer.py`, `tests/test_metrics.py`
  - Verify: output shapes correct, gradients finite, learning on synthetic signal, checkpoint save/restore, DM test accuracy
  - Report: pass/fail per test; debugging guidance if failures (e.g., "test_gradient_flow FAILED for iTransformer; check attention mechanism for NaN propagation")
  - _Requirements: Model validation, trainer reliability_

---

## Phase 9: Full Experiment Execution (3 tasks)

- [ ] 9.1 Run Experiment 1 (Horizon Sweep)
  - Execute `src/experiments/exp1_horizon.py`: train all 4 models × 4 horizons × 3 seeds = 48 runs
  - Resume if interrupted; track progress in run_manifest.json
  - Save: `outputs/results/exp1_results.json` (structured), `outputs/results/exp1_predictions.csv`
  - Report: completion status (X of 48 runs), failed runs (if any), total wall time, per-model wall time summary
  - _Requirements: Experiment 1 completion_

- [ ] 9.2 Run Experiment 2 (Lookback Sweep)
  - Execute `src/experiments/exp2_lookback.py`: train all 4 models × 4 lookbacks × 3 seeds = 48 runs
  - Compute threshold table: for each model, L_min where val_MSE ≤ best × 1.05
  - Save: `outputs/results/exp2_results.json`, `outputs/results/exp2_predictions.csv`, `outputs/results/threshold_table.json`
  - Report: completion status, threshold findings (e.g., "DLinear: L_min=192; NLinear: L_min=192; PatchTST: L_min=336; iTransformer: L_min=336")
  - _Requirements: Experiment 2 completion_

- [ ] 9.3 Run Experiment 3A & 3B
  - Execute `src/experiments/exp3_repr_shift.py`:
    - 3A: train 4 models × 2 branches × 3 seeds = 24 runs (reuse trained original-representation fits where possible)
    - 3B: evaluate on predefine test quarters (no additional training)
  - Save: `outputs/results/exp3a_results.json`, `outputs/results/exp3b_shift_analysis.json`
  - Report: completion status, representation findings (e.g., "Differencing helps NLinear by 5% MSE reduction; iTransformer unchanged"), shift findings
  - _Requirements: Experiment 3 completion_

---

## Phase 10: Report Generation & Publication (2 tasks)

- [ ] 10.1 Generate All Markdown Reports
  - Execute report generation pipeline: call all report generation functions (T7.1–T7.10) in sequence
  - Each report checks that required input data exists (e.g., `exp1_results.json` for T7.4)
  - Output: 9 reports in `outputs/reports/` (data_audit.md, eda_interpretation.md, methodology.md, exp1–3, statistical_analysis.md, computational_efficiency.md, reproducibility.md, final_report.md)
  - Verify: all reports have content; all plots linked correctly; all tables populated; no broken links
  - _Requirements: Report completeness_

- [ ] 10.2 Final Quality Assurance & Packaging
  - Validate all reports: Markdown syntax, broken links, missing plots, incomplete tables, dangling references
  - Verify artifact index: all checkpoints, configs, scalers, predictions CSVs present and accounted for
  - Create `outputs/reports/INDEX.md`: summary of all deliverables (9 reports, plot locations, result paths)
  - Package `outputs/`: compress to `.zip` or `.tar.gz` for archival and sharing
  - Final status report: "Study complete; all smoke tests passed; X of Y experiment runs succeeded; all reports generated; all data archived; reproducibility verified on [date/time]"
  - _Requirements: Quality, packaging, archival_

---

## Notes

- **Smoke Test Checkpoints**: T3.8, T8.1, T8.2 must pass before full experiment execution (T9.1–9.3)
- **Parallelization Opportunities**:
  - Phase 1: T1.2–1.4 can run in parallel (no dependencies)
  - Phase 3: T3.1–3.4 can run in parallel; all must complete before T3.5
  - Phase 4: T4.1, T4.2, T4.4, T4.6 can run in parallel
  - Phase 5: T5.2–5.4 can run in parallel once orchestrator (T5.1) queues runs
- **Sequential Bottlenecks**: T1.5 → T2.1; T3.8 → T5.1; T9.1–9.3 → T7.4–7.8; T7.1–7.10 → T10.1
- **Honest Reporting**: If any task fails, report exact error, data sample, and stopping point; do not fabricate or skip
- **Resume Capability**: All tasks save state (checkpoints, run_manifest.json); re-running X only completes incomplete portions
- **Language**: Python (inferred from workspace: `*.py` files, existing `clean_smartgrid_dataset.py`, `split_and_scale_smartgrid_data.py`)

---

## Task Dependency Graph

```json
{
  "waves": [
    {
      "id": 0,
      "tasks": [
        "1.1",
        "1.2"
      ]
    },
    {
      "id": 1,
      "tasks": [
        "1.3",
        "1.4"
      ]
    },
    {
      "id": 2,
      "tasks": [
        "1.5",
        "1.6"
      ]
    },
    {
      "id": 3,
      "tasks": [
        "1.7",
        "1.8"
      ]
    },
    {
      "id": 4,
      "tasks": [
        "2.1",
        "2.2"
      ]
    },
    {
      "id": 5,
      "tasks": [
        "2.3",
        "2.4"
      ]
    },
    {
      "id": 6,
      "tasks": [
        "2.5",
        "2.6"
      ]
    },
    {
      "id": 7,
      "tasks": [
        "3.1"
      ]
    },
    {
      "id": 8,
      "tasks": [
        "3.2",
        "3.3",
        "3.4"
      ]
    },
    {
      "id": 9,
      "tasks": [
        "3.5"
      ]
    },
    {
      "id": 10,
      "tasks": [
        "3.6",
        "3.7"
      ]
    },
    {
      "id": 11,
      "tasks": [
        "3.8"
      ]
    },
    {
      "id": 12,
      "tasks": [
        "4.1",
        "4.2",
        "4.4",
        "4.6"
      ]
    },
    {
      "id": 13,
      "tasks": [
        "4.3",
        "4.5"
      ]
    },
    {
      "id": 14,
      "tasks": [
        "5.1"
      ]
    },
    {
      "id": 15,
      "tasks": [
        "5.2",
        "5.3",
        "5.4",
        "5.5"
      ]
    },
    {
      "id": 16,
      "tasks": [
        "6.1"
      ]
    },
    {
      "id": 17,
      "tasks": [
        "6.2",
        "6.3",
        "6.4"
      ]
    },
    {
      "id": 18,
      "tasks": [
        "8.1",
        "8.2"
      ]
    },
    {
      "id": 19,
      "tasks": [
        "8.3",
        "8.4"
      ]
    },
    {
      "id": 20,
      "tasks": [
        "9.1",
        "9.2",
        "9.3"
      ]
    },
    {
      "id": 21,
      "tasks": [
        "7.1",
        "7.2",
        "7.3",
        "7.4",
        "7.5",
        "7.6",
        "7.7",
        "7.8",
        "7.9"
      ]
    },
    {
      "id": 22,
      "tasks": [
        "7.10"
      ]
    },
    {
      "id": 23,
      "tasks": [
        "10.1",
        "10.2"
      ]
    }
  ]
}
```

