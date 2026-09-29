# Comparative Forecasting Study: Technical Design

## Overview

This document specifies the technical architecture for a modular, reproducible comparative forecasting study on smart grid load prediction. The study compares four state-of-the-art time series models (DLinear, NLinear, PatchTST, iTransformer) across multiple experimental dimensions: horizon length, lookback window, stationarity representation (levels vs. differences), and temporal shift effects.

The design prioritizes:
- **Modularity**: Separated concerns (data, models, metrics, reporting)
- **Reproducibility**: Full configuration capture, deterministic seeding, artifact tracking
- **Data Governance**: No target leakage, scaler governance, chronological integrity
- **Rigor**: Validated preprocessing, comprehensive metrics, statistical testing with validity checks

---

## 1. Project Structure & Module Organization

```
comparative-forecasting-study/
├── configs/
│   ├── study.yaml                 # Master configuration (paths, L/H ranges, seeds, model params)
│   ├── models.yaml                # Model-specific architectures (dmodel, heads, layers, etc.)
│   └── experiments.yaml            # Experiment definitions (horizon, lookback, representation, shift)
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # CSV loading, timestamp parsing, schema validation
│   │   ├── audit.py               # Data audit: rows, coverage, duplicates, missing values
│   │   ├── preprocessor.py        # Scaling, differencing, gap handling, inversion
│   │   ├── splits.py              # Train/val/test chronological splits
│   │   └── windows.py             # Origin/window generation, stride-1 eligible origins
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract adapter: [B, L, C] → [B, H, C]
│   │   ├── dlinear.py             # DLinear from cure-lab/LTSF-Linear
│   │   ├── nlinear.py             # NLinear from cure-lab/LTSF-Linear
│   │   ├── patchtst.py            # PatchTST from yuqinie98/PatchTST
│   │   ├── itransformer.py        # iTransformer from thuml/iTransformer
│   │   └── trainer.py             # Training loop, early stopping, checkpoint logic
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── accuracy.py            # MAE, MSE per-lead
│   │   ├── dtw.py                 # Sakoe–Chiba constrained DTW
│   │   ├── diebold_mariano.py     # DM tests, HAC variance, Holm correction
│   │   └── profiling.py           # Training time, inference latency, params, FLOPs, memory
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # Run dispatcher, resume logic, deduplication
│   │   ├── exp1_horizon.py        # Experiment 1: horizon sweep (H ∈ {96,192,336,720})
│   │   ├── exp2_lookback.py       # Experiment 2: lookback sweep (L ∈ {96,192,336,720})
│   │   ├── exp3_repr_shift.py     # Experiment 3A/3B: representation & shift
│   │   └── baseline.py            # Persistence, daily/weekly naive
│   │
│   ├── plotting/
│   │   ├── __init__.py
│   │   ├── eda.py                 # Plot A–E + diagnostics
│   │   ├── results.py             # Experiment result plots, accuracy vs. cost
│   │   └── utils.py               # Color schemes, formatting, Markdown links
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── audit_report.py        # → data_audit.md
│   │   ├── eda_report.py          # → eda_interpretation.md
│   │   ├── methodology_report.py  # → methodology.md
│   │   ├── exp1_report.py         # → experiment_1_horizon.md
│   │   ├── exp2_report.py         # → experiment_2_lookback.md
│   │   ├── exp3_report.py         # → experiment_3_stationarity_shift.md
│   │   ├── stats_report.py        # → statistical_analysis.md
│   │   ├── compute_report.py      # → computational_efficiency.md
│   │   ├── repro_report.py        # → reproducibility.md
│   │   └── final_report.py        # → final_report.md (orchestrates all)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # YAML loading, validation, type hints
│       ├── logging.py             # Experiment logging, run status
│       ├── checksums.py           # MD5 dataset, preprocessing, config hashing
│       ├── device.py              # GPU/CPU detection, seed fixing
│       └── env_manifest.py        # Capture framework versions, OS, hardware
│
├── scripts/
│   ├── audit.py                   # CLI: python -m experiments.audit --config configs/study.yaml
│   ├── run.py                     # CLI: python -m experiments.run --config ... --experiment all --resume
│   ├── report.py                  # CLI: python -m experiments.report --results outputs/results ...
│   └── check_versions.py          # Dependency verification
│
├── tests/
│   ├── test_preprocessing.py      # Scaling round-trip, differencing, reconstruction
│   ├── test_splits.py             # Chronological correctness, no target leakage
│   ├── test_models.py             # Output shapes, gradients, tiny-signal learning
│   ├── test_metrics.py            # Hand-calculated MAE/MSE, DTW self=0
│   ├── test_baselines.py          # Seasonal naive no future reads
│   └── test_io.py                 # Resume, deduplication, artifact saving
│
├── outputs/
│   ├── runs/                      # Per-run: checkpoints, configs, scalers, curves
│   ├── predictions/               # CSV: origin, timestamp, lead, target, observed, pred, seed, model, repr
│   ├── reports/                   # All 9 Markdown reports
│   └── plots/                     # PNG/SVG/PDF figures
│
├── requirements.txt               # Pinned dependencies
├── setup.py                       # Package installation
└── README.md                      # Quick start, CLI reference, artifact index
```

---

## 2. Data Flow Architecture

### 2.1 Preprocessing Pipeline

The preprocessing pipeline follows a strict sequence to ensure data integrity and prevent target leakage:

1. **Load & Validate**
   - Read CSV with timestamp parsing and schema validation
   - Detect and handle timezone information
   - Validate row counts, data types, and required columns

2. **Audit & Document**
   - Compute audit statistics: total rows, temporal coverage, duplicates, missing values
   - Quantify cleaning changes (rows removed, values imputed)
   - Store audit artifacts for reproducibility

3. **Chronological Split**
   - Train: 2020–2023-01-01 (3 years)
   - Validation: 2023-01-01 – 2024-01-01 (1 year)
   - Test: 2024-01-01 onwards (held out)
   - Enforced via split boundaries; no reordering

4. **Scaler Fitting** (fit only on training)
   - Compute mean/std on training data only
   - Store scaler parameters (mean, scale) for reproducibility
   - Apply scaler to training, validation, and test

5. **Dual Representation Branches**
   - **Branch O (Original)**: scale training/val/test in original units
   - **Branch D (Differenced)**: compute differences (Δx[t] = x[t] - x[t-1]), then scale differences
   - Each branch maintains separate scaler parameters
   - Inverse transforms branch-specific (Branch D: integrate + unscale)

6. **Eligible Origin Generation**
   - Stride-1 origins within split boundaries
   - Constraint: origin + lookback ≤ split boundary, origin + lookback + horizon ≤ end-of-split
   - Generate origin manifests (metadata, indexing, boundaries)
   - Save for deterministic deduplication and cross-model consistency

7. **Artifact Preservation**
   - Save scalers (pickle)
   - Save origin manifests (JSON)
   - Save preprocessing configurations (JSON)
   - Compute and store checksums (MD5 of dataset, preprocessing settings)

### 2.2 Model Training Flow

**Input Format:**
- Shape: [batch_size, lookback, channels]
- Units: standardized (zero-mean, unit-variance) via scaler
- Branch-dependent: either original-scale or difference-scale

**Training Loop:**
1. Initialize model with config (lookback, horizon, channels, architecture params)
2. Fix random seeds (Python, NumPy, PyTorch, loader, device)
3. Create data loaders (batch_size=32, shuffle=True only for training)
4. Optimizer: Adam (LR=0.001, weight_decay=0.0, betas=(0.9, 0.999))
5. Loss function: MSE computed on standardized outputs (per-channel mean)
6. Validation metric: MSE computed on original-unit predictions (inverse transform applied)
7. Early stopping: monitor validation MSE, patience=7, restore best checkpoint
8. Max epochs: 50 (typical convergence earlier due to early stopping)
9. Gradient clipping: norm=1.0 to stabilize training
10. Precision: float32 (default PyTorch)

**Output:**
- Model checkpoint (state_dict)
- Optimizer state (for potential resumption)
- Training curves (CSV: epoch, train_loss, val_loss, learning_rate)
- Actual epochs trained
- Validation metadata (best epoch, best loss)

### 2.3 Evaluation & Prediction Flow

**Inference:**
- Load checkpoint and configuration
- Load scalers and origin manifest
- Batch inference: no_grad context, deterministic batch sizes
- Model output: [batch, horizon, channels] in standardized units

**Inverse Transform:**
- Branch O: apply scaler inverse (unscale from standardized to original)
- Branch D: apply scaler inverse to differences, then integrate differences back to levels

**Metric Computation:**
- Per-origin MAE, MSE, and per-lead breakdowns
- All metrics computed on original-unit predictions and observations
- Record prediction metadata: seed, model, representation, origin, lead

**Aggregation:**
- Per-seed: compute mean of metrics across all valid origins
- Across seeds: compute mean ± std of per-seed metrics
- Store full predictions CSV for post-hoc analysis

---

## 3. Model Adapter Pattern

### 3.1 Base Adapter Class

All models inherit from a canonical interface defined in `src/models/base.py`:

```python
class TimeSeriesAdapter(nn.Module):
    """Abstract base for canonical model adapters.
    
    Ensures consistent input/output shapes and interface across all models.
    """
    
    def __init__(self, lookback, horizon, channels, config):
        """
        Args:
            lookback: int, window length L
            horizon: int, forecast horizon H
            channels: int, number of channels (1 for univariate load, +extras for exog)
            config: dict, model-specific hyperparameters
        """
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.channels = channels
        self.config = config
    
    def forward(self, x):
        """
        Args:
            x: Tensor [batch, lookback, channels]
        
        Returns:
            Tensor [batch, horizon, channels]
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def get_flops(self, input_shape):
        """Compute FLOPs for forward pass at exact input shape.
        
        Args:
            input_shape: tuple (batch, lookback, channels)
        
        Returns:
            int, FLOPs count
        """
        pass
    
    def get_parameter_count(self):
        """Return total and trainable parameter counts."""
        pass
```

### 3.2 Canonical Model Implementations

**DLinear (src/models/dlinear.py)**
- Based on LTSF-Linear (Cure-Lab)
- Architecture: decomposition into trend (linear) and seasonal (moving average)
- Moving average kernel: 25 (standard, learned offset)
- Individual channel weights: False (shared linear layer)
- Key hyperparameters from config: d_model (not used; linear is univariate per channel)
- Output: [batch, horizon, channels]

**NLinear (src/models/nlinear.py)**
- Based on LTSF-Linear (Cure-Lab)
- Architecture: remove last value from input, apply linear layer, add back last value
- Individual channel weights: False (shared linear layer)
- Simpler than DLinear; often competitive
- Output: [batch, horizon, channels]

**PatchTST (src/models/patchtst.py)**
- Based on yuqinie98/PatchTST
- Architecture: patch embedding (patch_length=16, stride=8), multi-head attention
- Key hyperparameters:
  - d_model: embedding dimension (128)
  - n_heads: number of attention heads (4)
  - n_layers: transformer encoder layers (2)
  - d_ff: feed-forward dimension (256)
  - patch_length: 16 (fixed)
  - stride: 8 (fixed)
  - dropout: 0.1
- Channel independence: True (each channel processed separately)
- Output: [batch, horizon, channels]

**iTransformer (src/models/itransformer.py)**
- Based on thuml/iTransformer
- Architecture: inverted attention (variables as tokens, time as sequence dimension)
- Key hyperparameters:
  - d_model: 128
  - n_heads: 4
  - n_layers: 2
  - d_ff: 256
  - dropout: 0.1
- Learns cross-variable dependencies (contrasts with channel-independent models)
- Output: [batch, horizon, channels]

### 3.3 Shared Training Protocol

All models use an identical training configuration:

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Standard, adaptive learning rates |
| Learning Rate | 0.001 | Conservative; prevents divergence |
| Weight Decay | 0.0 | No regularization (experiment focus) |
| Batch Size | 32 | Balance memory and gradient stability |
| Max Epochs | 50 | Sufficient for typical convergence |
| Early Stopping Patience | 7 | Monitor validation loss, stop if no improvement |
| Gradient Clip Norm | 1.0 | Prevent exploding gradients |
| Precision | float32 | Default; sufficient for TS forecasting |
| Seed Fixing | Yes | Python, NumPy, PyTorch, device |

**Validation & Checkpointing:**
- Validation run every epoch (no train-val interleaving)
- Checkpoint metric: validation MSE in original units (after inverse transform)
- Best checkpoint: lowest validation MSE epoch
- Restore best weights at early stopping trigger

---

## 4. Experiment Orchestration

### 4.1 Experiment Dispatcher

The orchestrator (`src/experiments/orchestrator.py`) manages the full experiment lifecycle:

**Configuration Expansion:**
- Read `configs/experiments.yaml` and `configs/study.yaml`
- Expand experiment definitions into individual runs: one (L, H, representation, seed, model) tuple per run
- Total runs: 4 models × 4 lookbacks × 4 horizons × 2 representations × 3 seeds = 384 runs (for full grid)
- Experiments may define subsets (e.g., Exp 1 fixes L=336)

**Run Deduplication:**
- Compute run ID: hash(config, dataset, preprocessing settings, origin set, seed)
- Load prior `run_manifest.json` to find completed runs
- If ID in completed runs: skip training, load predictions from cache
- If ID not found: queue for training

**Dependency Management & Queueing:**
- Parse experiment definitions for dependencies
- Example: Exp 3A may reuse trained models from Exp 1 (same original-representation fits)
- Queue runs respecting dependencies; can parallelize independent runs

**Parallel Execution:**
- Concurrent limit: 5 runs (respects GPU memory, prevents thrashing)
- Per-run GPU allocation: auto-detect and distribute
- Fallback to CPU if GPU exhausted

**Resume Logic:**
- At startup: load `run_manifest.json` and identify pending/running/completed/failed runs
- Resume: skip completed runs, retry only incomplete runs (up to 3 attempts per run)
- At end: save updated manifest with newly completed runs and timestamps

**Reporting:**
- Print summary: total runs, completed, skipped (deduped), failed, pending
- Per-failed-run: exception name and message
- Total wall time

### 4.2 Experiment 1: Horizon Sweep

**Design:**
- Vary forecast horizon H across {96, 192, 336, 720} hours
- Fix lookback L = 336 hours
- Use original representation (Branch O)
- Three seeds: {42, 123, 2024}
- Four models: {DLinear, NLinear, PatchTST, iTransformer}

**Origin Constraint:**
- Eligible origins: those where origin + L + H ≤ split boundary
- Ensure valid training windows; varies by H (larger H → fewer origins)

**Metrics Computed:**
- Per-lead MAE and MSE (k ∈ [1, H])
- Per-model aggregates: mean ± std across origins, then across seeds
- DTW distance per-seed
- Per-lead analysis: errors increase with lead (typically); identify inflection points

**Output:**
- Exp 1 result table (H × model × seed matrix of MSE values)
- Per-lead accuracy plots (error vs. lead k)
- Accuracy-to-cost trade-off (MSE vs. H)

### 4.3 Experiment 2: Lookback Window Sweep

**Design:**
- Vary lookback window L across {96, 192, 336, 720} hours
- Fix horizon H = 192 hours
- Use original representation (Branch O)
- Three seeds, four models

**Threshold Selection:**
- For each L: compute validation MSE (per-seed mean)
- Find L_threshold: smallest L where val_MSE ≤ best_val_MSE × 1.05 (5% tolerance)
- Interpretation: minimum sufficient context window for stable performance

**Metrics Computed:**
- MAE, MSE per L and seed
- Training time, inference latency, parameter count, FLOPs per L
- Identify compute/accuracy Pareto frontier (sweet spots)

**Output:**
- Exp 2 result table (L × model × seed matrix)
- Threshold table: (model, L_threshold, val_MSE_best, L_sufficient)
- Accuracy vs. cost plots (MSE and latency vs. L)

### 4.4 Experiment 3A: Representation Comparison

**Design:**
- Compare Branch O (original units) vs. Branch D (differences) representations
- Fix L = 336, H = 192
- Fit separate models for each branch
- Stationarity assessment: ADF/KPSS tests on training data (both levels and differences)

**Stationarity Testing:**
- Applied to training data only (no lookahead)
- ADF test: reject null of unit root (non-stationary) if p < 0.05
- KPSS test: reject null of stationarity if p < 0.05
- Report: test statistic, p-value, critical values for α ∈ {0.01, 0.05, 0.10}

**Metrics Computed:**
- MSE and MAE per branch (both evaluated in original units for fair comparison)
- Reconstruction error: Branch D differences → integrate → compare to levels
- Per-lead breakdown to identify representation advantage at different lags

**Output:**
- Stationarity table (load levels and first differences; ADF/KPSS results)
- Accuracy comparison table (Branch O vs. D, per model and seed)
- Representation-specific insights (e.g., "differencing helps NLinear but not iTransformer")

### 4.5 Experiment 3B: Temporal Shift Effects

**Design:**
- Partition test year (2024) into quarters
- Predefine quarters before viewing error results (prevent p-hacking)
- Characterize load shift per quarter: mean, std, daily/weekly profiles
- Compare error metrics across quarters to quantify shift sensitivity

**Temporal Shift Quantification:**
- Load mean per quarter: measure absolute and relative change from training
- Load std per quarter: variability increase/decrease
- Daily profile: peak hour, peak magnitude, off-peak values
- Weekly effect: weekday/weekend separation, holiday effects (if identifiable)

**Error Attribution:**
- Model errors per quarter: MAE and MSE computed separately
- Correlation analysis: error increase vs. measured shift magnitude
- Identify which models are robust/sensitive to seasonal/structural shifts

**Output:**
- Shift table (quarter | mean_load | std_load | peak_hour | peak_magnitude | notes)
- Accuracy by quarter table (quarter × model × error_metric)
- Error-shift correlation analysis with visualization

---

## 5. Metrics Computation

### 5.1 Accuracy Metrics

**Mean Absolute Error (MAE):**
```
MAE = mean(|predicted[o,k] - observed[o,k]|)
    for all valid origin/lead pairs (o, k)
```
- Units: original (kW for load; explicitly labeled)
- Per-lead version: separate MAE for each k ∈ [1, H]
- Interpretation: average magnitude of errors in original units

**Mean Squared Error (MSE):**
```
MSE = mean((predicted[o,k] - observed[o,k])^2)
    for all valid origin/lead pairs (o, k)
```
- Units: original squared (kW² for load)
- Per-lead version: separate MSE for each k
- Interpretation: quadratic penalty emphasizes large errors

**Per-Lead Breakdown:**
- Compute MAE and MSE for each forecast lead k separately
- Visualize error growth with increasing lead time
- Identifies horizon-dependent model strengths/weaknesses

### 5.2 Dynamic Time Warping (DTW)

**Algorithm:**
- Sakoe–Chiba Band Constraint: ±24 hours (controls computational cost)
- Local cost: absolute error |predicted[k] - observed[k]| (or squared error)
- Allowed steps: (1,0), (0,1), (1,1) (fully flexible DTW)
- Endpoints: constrained (0,0) → (H-1, H-1)

**Tie-Breaking:**
- Deterministic ordering: prefer diagonal (1,1) > left (1,0) > down (0,1)
- Ensures reproducibility across implementations

**Path Normalization:**
- Raw DTW cost: cumulative distance along optimal path
- Normalized DTW: cumulative cost / path length
- Interpretation: average per-step distortion (scale-invariant)

**Sampling Strategy:**
- For large H: sample every 24 hours to reduce computational cost
- Record number of samples and report explicitly
- Note: subsampling may miss high-frequency misalignments

**Self-Distance Check:**
- DTW(x, x) must equal 0 (validation)
- Used in unit tests to verify implementation correctness

### 5.3 Diebold–Mariano Statistical Test

**Procedure:**
1. **Loss Differential:** d[t] = loss_A[t] - loss_B[t]
   - For each origin t, compute squared error at final lead k=H
   - d[t] quantifies relative performance
   
2. **Test Statistic:**
   ```
   DM = mean(d) / sqrt(Var_HAC(d))
   ```
   - Numerator: mean difference (positive if B outperforms A)
   - Denominator: heteroskedasticity-and-autocorrelation-consistent (HAC) variance

3. **HAC Variance Estimation:**
   - Kernel: Bartlett (common for time series)
   - Bandwidth: at least H-1 hours (to account for hourly forecast errors correlation)
   - Formula: γ[0] + 2 * Σ w[j] * γ[j] (Newey-West style)
   - Degenerate case: if Var_HAC ≈ 0, report as unsupported; flag for review

4. **Critical Region:**
   - Distribution: t-distribution with n-2 degrees of freedom (small-sample correction)
   - Two-tailed test: reject H0 (equal performance) if |DM| > t_{α/2, n-2}
   - Report: p-value, critical value, degrees of freedom

5. **Multiple Comparison Correction:**
   - Scenarios: 6 pairwise model comparisons per condition (4 choose 2)
   - Correction method: Holm–Bonferroni (less conservative than Bonferroni, maintains family-wise error rate)
   - Procedure: sort p-values, apply threshold α / (k+1-i) for i-th smallest
   - Report: original p-values, adjusted p-values, rejection threshold, number of rejected comparisons
   - **Caution flag**: emphasize sensitivity to family size (6 comparisons vs. typical 2-3)

6. **Validity Checks:**
   - Sample size: if n < 30, note small-sample assumption reliance
   - Stationarity: assume d[t] is stationary (should hold if both forecasts comparably biased)
   - HAC convergence: if bandwidth exhausts data, report and flag

7. **Sensitivity Analysis:**
   - Rerun DM with alternative seeds (e.g., seed=123 and seed=2024)
   - Report variance of DM statistic and p-value across seeds
   - If high variance, conclude sensitivity is data-dependent

8. **Reporting Template:**
   ```
   Model A vs. Model B (Experiment X, Lead k=H):
   - Mean Diff: +5.2 kW² (B better by 5.2)
   - DM Statistic: +2.15
   - p-value: 0.038
   - Adjusted p-value (Holm-6): 0.168 (not significant after correction)
   - HAC Variance: 5.87, Bandwidth: 192
   - Sample Size: n=487 origins
   - Seed Sensitivity: DM ∈ [2.10, 2.19] (low variance)
   ```

### 5.4 Computational Profiling Metrics

**Training Time:**
- Wall time from model initialization to final checkpoint save
- Includes data loading, backward passes, validation epochs
- Reported per seed; aggregated as mean ± std

**Inference Latency:**
- Batch size 1: single-sample prediction (typical edge use)
- Batch size 32: vectorized prediction (typical batch)
- Procedure: 20 warm-up iterations, 100 timed iterations, report median and p95
- Units: milliseconds

**Parameter Count:**
- Total parameters: all model weights and biases
- Trainable vs. frozen (typically all trainable for this study)
- Reported per model and architecture

**FLOPs (Floating-Point Operations):**
- Estimated via profiler (e.g., torchprofile, thop) for forward pass
- Shape-dependent: [batch=32, lookback, channels=1]
- Partial estimates labeled (e.g., if attention FLOPs uncertain, note "estimated")
- Tool and version recorded (e.g., "thop v0.1.2")

**Memory Usage:**
- Peak GPU memory during training (GB)
- Peak device memory during inference
- CPU memory if measurable
- Reported per seed; aggregate as max and mean

**Reporting Format:**
```
| Model | L | H | Time (s) | Latency_1 (ms) | Latency_32 (ms) | Params | FLOPs | Memory (GB) |
|-------|---|---|----------|----------------|-----------------|--------|-------|-------------|
| DLinear | 336 | 192 | 12.5±0.3 | 1.2±0.1 | 0.8±0.05 | 12K | 50M | 0.8 |
```

---

## 6. Statistical Testing & Validity

### 6.1 Assumptions & Checks

**Diebold–Mariano Assumptions:**
1. Loss differentials d[t] are stationary (should hold if both forecasts comparably distributed)
   - Check: plot d[t] over time; visually inspect for trends
2. HAC variance estimate is stable (Bartlett kernel convergence)
   - Check: verify bandwidth < n/2; compare HAC to standard variance
3. Sample size n adequate for t-distribution approximation
   - Rule: if n < 30, report degrees of freedom and note assumption reliance

### 6.2 Degeneracy Handling

**When HAC Variance is Near-Zero:**
- Possible cause: loss differentials are nearly identical (models are indistinguishable)
- Action: report as "statistically unsupported" rather than highly significant
- Investigate: compute range and quantiles of d[t]; if near-zero, confirm equivalence

**When Sample Size is Very Small:**
- If n < 10: DM test unreliable; switch to permutation test or Bootstrap CI
- Report explicitly: "Sample size insufficient for parametric DM test"

### 6.3 Sensitivity & Robustness

**Seed Variation:**
- Run DM tests with all three seeds independently
- Report: DM ∈ [min, max] across seeds
- Interpretation: if wide range, conclude effect is seed-dependent

**Lead Robustness:**
- Compute DM not just at k=H, but at k ∈ {H/4, H/2, 3H/4, H}
- Identify if effect is consistent across leads or concentrated at specific lags

**Temporal Stability:**
- Partition test set into quarters; compute DM within each quarter
- If effect flips across quarters, note sensitivity to temporal shift

### 6.4 Effect Size & Practical Significance

**Mean Difference:**
- Report in original units (e.g., "Model B produces 5.2 kW² lower MSE at k=H")
- Compare to baseline variance and typical operational thresholds

**Cohen's d (if applicable):**
- Effect size standardized by pooled standard deviation
- Interpretation: d < 0.2 (negligible), 0.2–0.8 (small), 0.8–∞ (large)

**Practical Context:**
- Relate to domain: "5.2 kW² improvement represents X% reduction in RMSE relative to mean forecasted load"
- Note if improvement is material to grid operations

---

## 7. Reproducibility & Checkpointing

### 7.1 Run Identification

**Unique Run ID:**
```
run_id = hash(
  config_yaml + 
  dataset_md5 + 
  preprocessing_settings + 
  origin_manifest_hash + 
  seed
)
```
- Ensures identical runs are detected and reused
- Stored in `run_manifest.json`

**Metadata Stored Per Run:**
```json
{
  "run_id": "abc123...",
  "timestamp": "2024-01-15T10:23:45Z",
  "model": "DLinear",
  "lookback": 336,
  "horizon": 192,
  "representation": "original",
  "seed": 42,
  "status": "completed",
  "epochs": 18,
  "val_mse_best": 2547.3,
  "train_time_seconds": 12.5,
  "artifact_paths": {
    "checkpoint": "outputs/runs/abc123.pt",
    "scalers": "outputs/runs/abc123_scalers.pkl",
    "curves": "outputs/runs/abc123_curves.csv",
    "config": "outputs/runs/abc123_config.json"
  }
}
```

### 7.2 Checkpoint Contents

**Per-Run Checkpoint Directory (outputs/runs/{run_id}/):**
- `model_state.pt`: PyTorch model state_dict
- `optimizer_state.pt`: Adam state (for potential resumption)
- `config.json`: Exact configuration used
- `scalers.pkl`: StandardScaler objects (mean, scale) for each branch
- `training_curves.csv`: epoch, train_loss, val_loss, learning_rate
- `metadata.json`: epochs, best_epoch, best_val_loss, total_time
- `environment.json`: Python version, PyTorch version, device, OS, hardware

### 7.3 Resume Logic

**Startup Procedure:**
1. Load `run_manifest.json` (created if missing)
2. For each queued run (from config expansion):
   - Compute run_id
   - Check if run_id exists in manifest
   - If exists and status = "completed": load predictions, skip training
   - If exists and status ≠ "completed": mark for retry (up to 3 attempts)
   - If not exists: add to manifest with status = "pending"

**Run Status Lifecycle:**
```
pending → running → completed
       ↘ failed (up to 3 retries) ↘ abandoned
```

**Deduplication Caveats:**
- If config matches but data changes (e.g., new raw CSV rows), recompute (detect via dataset_md5)
- If preprocessing settings change, recompute (detect via preprocessing_settings hash)
- If origin manifest changes, recompute (detect via origin_manifest_hash)

### 7.4 Validation Checklist

**Before Full Run:** Execute these validation tests:

1. **Chronological Correctness:**
   - Verify split boundaries: train < val < test chronologically
   - Verify no origin is outside its split
   - Verify no target leakage: all origins have lookback + horizon ≤ split boundary

2. **Scaler Governance:**
   - Verify scaler fitted only on training data
   - Verify scaler not refitted on val/test
   - Verify inverse transform correctly reconstructs original units

3. **Differencing Round-Trip (Branch D Only):**
   - Forward: original → difference → scale
   - Backward: unscale → integrate → verify original values
   - Tolerance: < 1e-6 relative error

4. **Origin Consistency:**
   - Generate origins for two models (e.g., DLinear and NLinear)
   - Verify origin sets identical (same indices, boundaries)
   - Verify models see identical training/val windows

5. **Model Output Shapes:**
   - Forward pass on dummy input: [32, L, C] → output [32, H, C]
   - Verify no internal transpositions or squeezing

6. **Gradient Flow:**
   - Verify backward pass succeeds (loss.backward())
   - Verify no NaN/Inf gradients
   - Verify gradient magnitudes reasonable (order 1e-4 to 1e-1)

7. **Tiny-Signal Learning:**
   - Create synthetic signal: constant + small noise
   - Train model to near-zero loss
   - Verify model is trainable and not fundamentally broken

8. **Baseline Integrity (Seasonal Naive):**
   - Verify seasonal naive does not read future seasonal lags
   - Verify predictions use only historical same-day patterns
   - Verify no lookahead

9. **DTW Self-Distance:**
   - Compute DTW(x, x) for random predictions
   - Verify result = 0 (or < 1e-9 if numerical precision is concern)

10. **DM Test Degeneracy:**
    - Run DM on two identical loss vectors
    - Verify p-value ≈ 1 (no difference detected)
    - Verify no exceptions or NaN values

---

## 8. Report Generation Pipeline

### 8.1 Report Sequence & Dependencies

All reports are generated from saved artifacts (never computed from raw data live):

```
1. audit_report.py     → loads audit artifacts    → data_audit.md
2. eda_report.py       → loads plots, curves      → eda_interpretation.md
3. methodology_report.py → loads configs, splits  → methodology.md
4. exp1_report.py      → loads Exp 1 predictions  → experiment_1_horizon.md
5. exp2_report.py      → loads Exp 2 predictions  → experiment_2_lookback.md
6. exp3_report.py      → loads Exp 3 predictions  → experiment_3_stationarity_shift.md
7. stats_report.py     → loads all predictions    → statistical_analysis.md
8. compute_report.py   → loads profiling data     → computational_efficiency.md
9. repro_report.py     → loads run_manifest       → reproducibility.md
10. final_report.py    → orchestrates 1-9         → final_report.md
```

### 8.2 Individual Reports

**1. data_audit.md**
- Source: audit artifacts (JSON with row counts, coverage, duplicates, missing values)
- Sections:
  - Data source and loading procedure
  - Row counts before/after cleaning
  - Temporal coverage: start, end, gaps
  - Duplicates and missing values: location, count
  - Cleaning changes: rows removed, imputations applied
  - Conclusion: fitness for forecasting

**2. eda_interpretation.md**
- Source: plots A–E, diagnostic statistics
- Sections:
  - Plot A: entire 5-year time series with trends
  - Plot B: one-week detail (intra-day patterns)
  - Plot C: one-month detail (weekly patterns)
  - Plot D: daily/weekly profiles (aggregated statistics)
  - Plot E: heatmap of weekday × hour (cyclicity)
  - Text interpretation: key patterns, seasonality, anomalies
  - Implications for modeling (stationarity, lag structure, horizon-dependent behavior)

**3. methodology.md**
- Source: config files, split info, baseline definitions
- Sections:
  - Data splits: train/val/test boundaries, row counts per split
  - Origin generation: stride-1 logic, eligible L/H constraints
  - Preprocessing: scaling, differencing branches
  - Baselines: persistence (last value), daily/weekly naive
  - Models: architecture summaries, hyperparameters
  - Training protocol: optimizer, loss, early stopping, seed fixing
  - Evaluation: metrics, statistical testing

**4. experiment_1_horizon.md**
- Source: Exp 1 predictions CSV
- Sections:
  - Design: H ∈ {96, 192, 336, 720}; L=336
  - Results: table (H × model × seed × {MAE, MSE, DTW})
  - Per-lead analysis: error vs. lead k; plots
  - Findings: accuracy degradation rate, inflection points
  - Model ranking: model performance across horizons

**5. experiment_2_lookback.md**
- Source: Exp 2 predictions CSV
- Sections:
  - Design: L ∈ {96, 192, 336, 720}; H=192
  - Threshold results: L_threshold per model
  - Accuracy results: table (L × model × seed × {MAE, MSE})
  - Computational cost: training time, parameters, FLOPs, latency vs. L
  - Pareto frontier: models on accuracy-latency trade-off curve
  - Recommendation: practical L values (if L_threshold < L_max, use L_threshold)

**6. experiment_3_stationarity_shift.md**
- Source: Exp 3A/3B predictions, stationarity tests
- Sections:
  - **3A – Representation:**
    - Stationarity tests: ADF/KPSS on levels and differences
    - Accuracy comparison: Branch O vs. Branch D (per model, seed)
    - Reconstruction analysis: differencing round-trip error
    - Conclusion: which representation preferred by each model
  - **3B – Temporal Shift:**
    - Load characteristics per test quarter: mean, std, daily profile
    - Errors per quarter: table (quarter × model × {MAE, MSE})
    - Correlation: error increase vs. measured shift magnitude
    - Robust vs. sensitive models: identification

**7. statistical_analysis.md**
- Source: all predictions, DM test results
- Sections:
  - DM test setup: loss metric, correction method
  - Results: table of pairwise comparisons (Model A vs. Model B; DM, p-value, adj. p-value)
  - Sensitivity: DM across seeds, leads, quarters (if variance high, note)
  - Conclusions: significant differences, practical significance, limitations

**8. computational_efficiency.md**
- Source: profiling data (timing, params, FLOPs, memory)
- Sections:
  - Training efficiency: time per epoch, total time vs. model
  - Inference latency: batch size 1 and 32 (median, p95)
  - Model size: parameters, FLOPs (methodology noted: profiler tool/version)
  - Memory footprint: peak device and CPU memory
  - Hardware specification: OS, GPU/CPU model, RAM
  - Trade-offs: visualize accuracy vs. latency, accuracy vs. params

**9. reproducibility.md**
- Source: run manifest, environment snapshots
- Sections:
  - CLI commands: exact commands to reproduce study
  - Environment: Python version, PyTorch, CUDA, OS
  - Dependencies: pinned library versions (from requirements.txt)
  - Run status: completed, deduped, failed runs
  - Artifact locations: where all outputs saved
  - Data integrity: dataset MD5, preprocessing MD5, config MD5
  - Resumption instructions: how to retry failed runs

**10. final_report.md**
- Orchestrates all reports; synthesizes findings
- Sections:
  - Executive summary (1-2 pages: key findings, recommendations)
  - Study overview (goals, methodology summary)
  - Data characteristics (reference to eda_interpretation.md)
  - Model comparison across experiments (summary tables)
  - Recommendations: model choice, lookback window, representation
  - Limitations and future work
  - References and reproducibility links

### 8.3 Output Formatting

**Markdown Conventions:**
- All numeric values include units (kW, kW², hours, ms, GB, M for millions)
- Links to plots: relative paths, e.g., `![Plot](../plots/plot_a.png)`
- Tables: model | representation | L | H | MAE | MSE | DTW | latency | params | FLOPs | memory
- Captions include measured values: "Peak load at 18:00±2 hours; weekday avg 65 MW; weekend avg 48 MW"

**Color Scheme (Consistent Across Plots):**
- DLinear: blue (#1f77b4)
- NLinear: orange (#ff7f0e)
- PatchTST: green (#2ca02c)
- iTransformer: red (#d62728)
- Baselines: gray (#7f7f7f)

**File Formats:**
- Predictions: CSV (origin, timestamp, lead, target, observed, pred, seed, model, repr)
- Plots: PNG (inline in Markdown), SVG/PDF (for publication)
- Results: JSON and CSV for downstream analysis

---

## 9. Configuration Management

### 9.1 configs/study.yaml

```yaml
# Master configuration file
dataset:
  path: DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv
  timestamp_col: timestamp
  target_col: load_kw
  channels: [temperature]  # Additional exogenous variables
  
splits:
  train_start: "2020-01-01"
  train_end: "2023-01-01"
  val_end: "2024-01-01"
  # test_end: open (all data from 2024-01-01 onwards)

lookbacks: [96, 192, 336, 720]
horizons: [96, 192, 336, 720]
seeds: [42, 123, 2024]

training:
  batch_size: 32
  max_epochs: 50
  early_stopping_patience: 7
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0
  gradient_clip_norm: 1.0
  precision: float32

models:
  dlinear:
    d_model: 128  # Unused for linear models
    moving_avg_kernel: 25
    individual_channel_weights: false
  
  nlinear:
    d_model: 128
    individual_channel_weights: false
  
  patchtst:
    d_model: 128
    n_heads: 4
    n_layers: 2
    d_ff: 256
    patch_length: 16
    stride: 8
    dropout: 0.1
    channel_independent: true
  
  itransformer:
    d_model: 128
    n_heads: 4
    n_layers: 2
    d_ff: 256
    dropout: 0.1

output_dir: outputs
device: auto  # auto-detect GPU; fallback CPU
```

### 9.2 configs/experiments.yaml

```yaml
experiments:
  exp1_horizon:
    name: "Experiment 1: Horizon Sweep"
    lookback: 336
    horizons: [96, 192, 336, 720]
    representation: original
    models: [dlinear, nlinear, patchtst, itransformer]
    seeds: [42, 123, 2024]
  
  exp2_lookback:
    name: "Experiment 2: Lookback Window"
    lookbacks: [96, 192, 336, 720]
    horizon: 192
    representation: original
    models: [dlinear, nlinear, patchtst, itransformer]
    seeds: [42, 123, 2024]
  
  exp3a_representation:
    name: "Experiment 3A: Representation (Original vs. Differenced)"
    lookback: 336
    horizon: 192
    representations: [original, differenced]
    models: [dlinear, nlinear, patchtst, itransformer]
    seeds: [42, 123, 2024]
  
  exp3b_shift:
    name: "Experiment 3B: Temporal Shift Effects"
    lookback: 336
    horizon: 192
    representation: original
    models: [dlinear, nlinear, patchtst, itransformer]
    seeds: [42, 123, 2024]
    # Reuse trained models from exp1 or exp3a
    dependency: exp1_horizon
```

### 9.3 CLI Usage

```bash
# Audit data
python -m experiments.audit --config configs/study.yaml

# Run all experiments
python -m experiments.run --config configs/study.yaml --experiment all --resume

# Run specific experiment
python -m experiments.run --config configs/study.yaml --experiment exp1_horizon --resume

# Generate reports
python -m experiments.report --results outputs/results --output outputs/reports/final_report.md

# Check environment and dependencies
python scripts/check_versions.py
```

---

## 10. Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable requirements and machine-verifiable correctness guarantees.*

### Property 1: Chronological Data Integrity

*For any* train/validation/test split boundaries, all training origins SHALL occur before all validation origins, and all validation origins SHALL occur before all test origins. No target leakage: for any origin at index o with lookback L and horizon H, the values at indices [o+L, o+L+H) SHALL be strictly after o+L in temporal order.

**Validates: Reproducibility & Data Governance Requirements**

### Property 2: Scaler Fit-Transform Governance

*For any* dataset split, the StandardScaler SHALL be fitted exclusively on training data. When applied to validation or test data, the scaler SHALL use only the training mean and scale; it SHALL NOT recompute or update parameters based on validation/test values.

**Validates: Reproducibility & Preprocessing Integrity Requirements**

### Property 3: Model Output Shape Consistency

*For any* model instance with specified lookback L, horizon H, and channels C, and *for any* valid input batch of shape [B, L, C], the model forward pass SHALL produce output of shape [B, H, C]. The output SHALL be in standardized units (zero-mean, unit-variance) matching the input representation branch (original or differenced).

**Validates: Model Adapter Pattern & Training Requirements**

### Property 4: Preprocessing Round-Trip (Differencing Branch)

*For any* sequence in the differencing branch, applying the forward transformation (compute differences, scale) followed by the inverse transformation (unscale, integrate) SHALL recover the original sequence within numerical precision (relative error < 1e-6).

**Validates: Preprocessing Pipeline Integrity**

### Property 5: Origin Eligibility Constraint

*For any* lookback L and horizon H, an origin at index o is eligible if and only if:
- o ≥ L (sufficient history available)
- o + L + H ≤ split_boundary (complete forecast window within split)
All origins returned by the origin generator SHALL satisfy these constraints; no origin SHALL violate these boundaries.

**Validates: Origin Generation & Window Consistency**

### Property 6: Seed Reproducibility

*For any* fixed configuration (model, L, H, representation, dataset, preprocessing settings) and *for any* two runs with identical seed, the trained model checkpoint, validation loss trajectory, and predictions SHALL be identical (or differ only by numerical precision < 1e-6).

**Validates: Reproducibility & Deterministic Seeding**

### Property 7: Inverse Transform Correctness

*For any* prediction in standardized units and *for any* representation branch (original or differenced), applying the branch-specific inverse transform SHALL yield a prediction in original units. For Branch O, unscaling SHALL apply. For Branch D, unscaling + integrating differences SHALL apply. The result SHALL match the original-scale prediction from an equivalent Branch O model within numerical precision.

**Validates: Metric Computation Validity & Fair Model Comparison**

### Property 8: Baseline No-Lookahead

*For any* seasonal naive baseline with specified season length s, the prediction at index t SHALL depend only on historical observations at index t-s (or earlier). The baseline SHALL NOT access any observations at indices [t, t+H).

**Validates: Baseline Integrity & Fair Comparison**

### Property 9: DTW Self-Distance

*For any* predicted sequence and *for any* observed sequence, the DTW distance DTW(sequence, sequence) SHALL equal zero (or < 1e-9 if numerical precision is a concern). This property validates the DTW implementation.

**Validates: Metric Correctness & Test Suite Validity**

### Property 10: Run Deduplication Consistency

*For any* configuration C that produces the same run_id as a prior completed run, reusing the completed run's predictions and model checkpoint SHALL produce identical results to re-training and re-evaluating. Formally: load(completed_run_artifact) == load(re_run_artifact) for identical config_hash, data_hash, preproc_hash, origin_hash, seed.

**Validates: Reproducibility & Resume Logic**

### Property 11: Diebold–Mariano Validity

*For any* identical loss vectors (loss_A == loss_B for all origins), the Diebold–Mariano test SHALL return a two-tailed p-value ≥ 0.95 (no significant difference detected). The test SHALL not produce NaN, Inf, or raise exceptions on valid input.

**Validates: Statistical Testing Validity & Implementation Correctness**

### Property 12: Configuration Consistency Across Models

*For any* two models (A and B) run on identical experiment settings (L, H, representation, seed, dataset), the set of eligible origins generated for model A SHALL be identical to the set for model B. This ensures fair cross-model comparison.

**Validates: Fair Model Comparison & Origin Consistency**

---

## 11. Testing Strategy

### Unit Tests
- **test_preprocessing.py**: Scaling round-trip, differencing, reconstruction, scaler governance
- **test_splits.py**: Chronological correctness, boundary enforcement, no target leakage
- **test_models.py**: Output shapes, gradient flow, tiny-signal learning
- **test_metrics.py**: Hand-calculated MAE/MSE, DTW self-distance, DM validity
- **test_baselines.py**: No-lookahead verification, seasonal-lag correctness
- **test_io.py**: Resume, deduplication, artifact loading

### Integration Tests
- End-to-end run with all components: data → preprocess → train → eval → report
- Verify artifact persistence and resume capability

### Validation Tests (Pre-Run Checklist)
- All 10 checks listed in Section 7.4 executed and passing before full experiment suite

---

## 12. Dependencies & Environment

| Dependency | Version | Purpose |
|---|---|---|
| PyTorch | ≥ 2.0 | Deep learning framework |
| scikit-learn | ≥ 1.0 | Scaling, metrics, preprocessing |
| pandas | ≥ 1.3 | Data loading, time handling |
| numpy | ≥ 1.20 | Numerical operations |
| matplotlib | ≥ 3.5 | Plotting, visualization |
| LTSF-Linear | (pinned commit) | DLinear, NLinear implementations |
| PatchTST | (pinned commit) | PatchTST implementation |
| iTransformer | (pinned commit) | iTransformer implementation |
| statsmodels | ≥ 0.13 | ADF/KPSS stationarity tests |
| PyYAML | ≥ 5.4 | Configuration file parsing |

**Environment Capture:**
- Python version, PyTorch/TensorFlow version, CUDA version (if applicable)
- OS, hardware (CPU model, GPU model, RAM)
- Captured at runtime and saved to `environment.json` per run

---

## 13. Summary

This design establishes a rigorous, modular architecture for comparative forecasting research. Key principles:

1. **Modularity**: Clear separation of concerns (data, models, metrics, reporting)
2. **Reproducibility**: Full configuration capture, deterministic seeding, artifact tracking, resume capability
3. **Data Governance**: Chronological integrity, scaler fit governance, no target leakage
4. **Rigor**: Validated preprocessing, comprehensive metrics (accuracy, DTW, computational), statistical testing with validity checks
5. **Scalability**: Parallel run dispatcher, efficient checkpointing, deduplication logic

The study will produce:
- 384 trained models (4 models × 4 L × 4 H × 2 representations × 3 seeds, accounting for shared baselines)
- Comprehensive predictions CSV with metadata
- 10 detailed Markdown reports synthesizing findings
- Run artifacts (checkpoints, scalers, curves) enabling full reproducibility

All statistical testing includes validity checks, sensitivity analysis, and caution flags for interpretation. Reports are generated from artifacts, never manually computed, ensuring traceability from raw data to conclusions.

