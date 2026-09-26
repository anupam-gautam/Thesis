# Smart-Grid Data Workflow

This guide explains how to run the Python scripts when starting with only the raw smart-grid CSV file.

Run every command from the project root (the folder containing `DataSet`, `myenv`, and the Python files).

## Required input

Place the raw data file at:

```text
DataSet/RAW_smartgrid_5yr_hourly_natural.csv
```

The file must contain a `timestamp` column, as well as the smart-grid columns used by the cleaning script.

## One-time environment setup

The project virtual environment needs pandas, scikit-learn, matplotlib, and NumPy. If the `myenv` folder already exists, activate it:

```powershell
.\myenv\Scripts\Activate.ps1
```

If PowerShell prevents activation, run the commands below without activating it by using `myenv`'s Python executable.

## Required processing order

1. Optional: inspect the raw data.
2. Clean and validate the raw data.
3. Split the cleaned data by time and scale the two continuous columns.
4. Optional: create exploratory plots.

### 1. Inspect the raw data (optional)

```powershell
.\myenv\Scripts\python.exe basic_data_overview.py
```

This reads the raw CSV and prints its shape, columns, data types, missing values, time range, and summary statistics. It does not create or change any data file.

### 2. Clean the raw data (required)

```powershell
.\myenv\Scripts\python.exe clean_smartgrid_dataset.py
```

This script parses and sorts timestamps, removes duplicate timestamps, fills gaps in the hourly timeline, handles missing values, and keeps the modelling columns. It produces:

```text
DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv
```

### 3. Split and scale the cleaned data (required)

```powershell
.\myenv\Scripts\python.exe split_and_scale_smartgrid_data.py
```

This script reads the cleaned CSV from step 2, makes chronological train/validation/test data sets, and prints the statistics learned from the training data. It produces:

```text
DataSet/splits/train_scaled.csv
DataSet/splits/validation_scaled.csv
DataSet/splits/test_scaled.csv
```

Read [04_train_validation_test_split_and_scaling.md](04_train_validation_test_split_and_scaling.md) for a detailed explanation of this step.

### 4. Create exploratory plots (optional)

```powershell
.\myenv\Scripts\python.exe load_time_series_analysis.py
```

This is an exploratory analysis script. It currently reads the raw CSV and saves plots in `plots/`. It is not required to create the cleaned or split data sets.

## Quick run: required steps only

If you have only raw data and want the three model-ready splits, run these two commands in order:

```powershell
.\myenv\Scripts\python.exe clean_smartgrid_dataset.py
.\myenv\Scripts\python.exe split_and_scale_smartgrid_data.py
```

## Expected split output

For the supplied five-year data set, the split script reports:

| Split | Period | Expected rows |
| --- | --- | ---: |
| Training | 2020-01-01 through 2022-12-31 | 26,304 |
| Validation | 2023-01-01 through 2023-12-31 | 8,760 |
| Test | 2024-01-01 onward | 8,784 |

If your data covers different dates, the row counts may differ. The three date boundaries still apply.
