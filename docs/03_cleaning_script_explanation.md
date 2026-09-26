# Cleaning Script Explanation

## Purpose

The cleaning script was created to process the raw smart-grid dataset in a reproducible and transparent way. The script performs the key time-series ETL tasks needed to prepare the dataset for multivariate analysis and forecasting without modifying the original source file.

---

## Script workflow

### 1. Load raw CSV

The script reads the raw dataset from the `DataSet` folder into a pandas DataFrame. This is the initial stage of the ETL pipeline and allows the data to be inspected and transformed in a controlled manner.

### 2. Validate timestamp presence

The script checks whether the `timestamp` column exists. If the timestamp column is missing, the pipeline stops immediately because time-series analysis cannot proceed without a valid temporal index.

### 3. Convert timestamp to datetime

The raw timestamp values are converted to pandas datetime objects using `pd.to_datetime()`. This is essential for chronological operations and date-based analysis.

### 4. Remove invalid timestamps

Any rows with invalid datetime values are removed. This prevents malformed rows from interrupting sorting, indexing, or resampling operations.

### 5. Sort data chronologically

The dataset is sorted in ascending order by `timestamp`. This ensures that the time series is in the correct sequence before any analysis or modelling is performed.

### 6. Detect and remove duplicate timestamps

The script identifies duplicate timestamps and removes them. Duplicate timestamps can distort time-series analysis and should not be present in a clean hourly time series.

### 7. Check for missing hourly intervals

The script creates a complete hourly datetime index between the minimum and maximum timestamps and compares it with the timestamps already present in the dataset. If any hours are missing, the dataset is reindexed to restore the full hourly structure.

### 8. Handle missing values

The script checks for null values across the dataset and fills them using time-aware approaches. Numeric variables are interpolated, while non-numeric variables are filled forward and backward as needed.

### 9. Select relevant variables

The script reduces the dataset to a subset of variables that are relevant for multivariate forecasting. The cleaned dataset keeps the target variable (`load_kw`) and the main exogenous drivers available at prediction time.

### 10. Save cleaned output

The final cleaned data is stored in a separate file so the original raw dataset remains unchanged. This is important for reproducibility and transparent research practice.

---

## Why this approach is appropriate

This procedure is standard for ETL in time-series data analysis because it ensures that the dataset is valid, complete, ordered, and model-ready. In this case, the dataset was already high quality, so the cleaning process mainly validated the integrity of the data and selected an efficient feature set without resorting to aggressive cleaning or heavy imputation.

---

## Final result

The cleaned dataset contains one hourly observation per time point with no duplicate or missing timestamps, no missing observations, and a compact multivariate structure suitable for model development and analysis.
