# Data Preprocessing and Cleaning for the Smart Grid Hourly Load Dataset

## 1. Dataset description

The dataset used in this study is an hourly smart-grid electricity load dataset covering a five-year period. It contains timestamped electricity demand records together with several exogenous variables such as temperature, rain-day indicators, holiday information, lockdown status, and related contextual features.

The raw file is stored as:

- `DataSet/RAW_smartgrid_5yr_hourly_natural.csv`

The cleaned version produced for modelling and analysis is stored as:

- `DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv`

---

## 2. Objective of preprocessing

The objective of the preprocessing stage was to ensure that the dataset met the expected requirements for time-series analysis and multivariate modelling. Specifically, the workflow focused on:

- validating the timestamp field,
- ensuring chronological ordering,
- removing duplicate entries,
- checking for missing timestamps and gaps,
- handling missing observations where necessary,
- selecting a relevant subset of variables for downstream modelling,
- preserving the raw dataset untouched while generating a cleaned analytical version.

---

## 3. Data-cleaning workflow

### Step 1: Load raw dataset

The raw CSV file was loaded using pandas. This step provided the initial structure of the dataset, including the column names, row counts, and data types. The raw file was not edited directly; instead, a separate cleaned dataset was generated.

### Step 2: Parse timestamp

The `timestamp` column was converted from a string representation into a pandas datetime format using `pd.to_datetime()`. This conversion was essential because time-series analysis depends on accurate datetime handling for sorting, indexing, resampling, and temporal feature extraction.

Any rows with invalid timestamp values were identified and removed to prevent corrupted time points from entering the analysis pipeline.

### Step 3: Sort chronologically

After parsing the timestamp, the records were sorted in ascending chronological order. Sorting is necessary for reliable time-series analysis because models and aggregate computations assume a consistent temporal sequence.

### Step 4: Check for duplicate timestamps

The dataset was checked for duplicate timestamp entries using the duplicated() function. If duplicate timestamps had been present, the last occurrence would be retained and earlier duplicates removed. In the current dataset, no duplicate timestamps were found, so no row removal was required at this step.

### Step 5: Check for missing timestamps

The full expected hourly time index was generated from the minimum timestamp to the maximum timestamp using hourly frequency. This expected range was compared against the timestamps present in the dataset.

The purpose of this step was to detect any missing hours or gaps in the temporal sequence. A continuous hourly index is critical for load forecasting and time-series modelling because missing gaps can distort seasonality, trend estimation, and model training.

In this dataset, the full hourly sequence was complete, and no missing timestamps were detected.

### Step 6: Handle missing observations

The dataset was inspected for null or missing values across all columns. Missing observations were handled using a time-aware imputation approach:

- numeric variables were interpolated using time-based interpolation,
- non-numeric fields were filled using forward/backward filling when needed.

This strategy preserves the continuity of the time series while avoiding arbitrary deletion of rows. In the current dataset, no missing observations were found, so this step had no effect.

### Step 7: Select variables

A final variable-selection step was applied to retain only the variables most relevant to multivariate forecasting and analysis. The cleaned dataset retained:

- `timestamp`
- `temperature`
- `is_rain_day`
- `holiday_type_national`
- `holiday_type_religious`
- `is_lockdown`
- `load_kw`

The original raw file also contained derived calendar fields such as `hour`, `weekday`, `month`, `is_weekend`, and seasonal indicators. These fields were not retained in the cleaned version because they are either directly derivable from the timestamp or considered redundant for the selected modelling configuration.

This variable selection reduces the dimensionality of the dataset while preserving strong predictive information for electricity-load analysis.

### Step 8: Save cleaned dataset

The cleaned output was written to a separate CSV file instead of modifying the original raw dataset. This preserves data provenance and ensures that the original source remains intact for future reprocessing or reproducibility checks.

The cleaned file generated was:

- `DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv`

---

## 4. Final data quality assessment

After preprocessing, the dataset was verified to contain:

- 43,848 rows,
- 7 columns,
- no duplicate timestamps,
- no missing timestamps,
- no missing observations,
- a complete hourly timeline spanning the full study period.

This indicates that the raw dataset was already structurally clean and did not require heavy repair before modelling. The preprocessing primarily served as a validation, standardization, and feature-selection stage rather than a large-scale data-cleaning intervention.

---

## 5. Interpretation for research reporting

The preprocessing stage was essential to ensure that the dataset satisfied the assumptions of time-series modelling. The key conclusion is that the original dataset was of high quality in terms of completeness and temporal integrity. As a result, the main data-processing effort was focused on standardizing timestamps, confirming continuity, and selecting a compact set of meaningful variables for multivariate modelling rather than imputing or removing large portions of the data.

This is important for reproducibility because it demonstrates that the cleaned dataset was generated through a transparent and defensible ETL procedure while preserving the integrity of the original observations.
