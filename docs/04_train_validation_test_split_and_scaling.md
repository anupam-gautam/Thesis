# Train, Validation, Test Split and Scaling

This document describes what `split_and_scale_smartgrid_data.py` does and why it is done in this order.

## Input and output

The script reads the cleaned data file:

```text
DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv
```

It writes three new CSV files:

```text
DataSet/splits/train_scaled.csv
DataSet/splits/validation_scaled.csv
DataSet/splits/test_scaled.csv
```

The original cleaned file is not changed.

## 1. Split the data by time

Smart-grid load data is time-series data, so it must stay in chronological order. The script does not shuffle rows.

| Data set | Start date | End rule | Purpose |
| --- | --- | --- | --- |
| Training | 2020-01-01 | Before 2023-01-01 | Used to train a model. |
| Validation | 2023-01-01 | Before 2024-01-01 | Used to compare model choices and settings. |
| Test | 2024-01-01 | All later data | Used once for final evaluation. |

Using past data to predict later data reflects the real forecasting situation.

## 2. Learn scaling values from training data only

The script standardizes only these continuous numeric columns:

```python
NUMERIC_COLUMNS = ["temperature", "load_kw"]
```

It creates a `StandardScaler` and fits it on the training set only:

```python
scaler = StandardScaler()
scaler.fit(training_data[NUMERIC_COLUMNS])
```

Fitting calculates the training average (`scaler.mean_`) and training standard deviation (`scaler.scale_`) for each of these columns.

For the supplied data, the printed training averages are approximately:

| Column | Training average |
| --- | ---: |
| `temperature` | 30.8372 |
| `load_kw` | 82,552.6941 |

These values come only from 2020 through 2022. Validation and test rows do not affect them.

## 3. Apply the same scaler to every split

After fitting, the script uses the same scaler for training, validation, and test data:

```python
scaled_training[NUMERIC_COLUMNS] = scaler.transform(training_data[NUMERIC_COLUMNS])
scaled_validation[NUMERIC_COLUMNS] = scaler.transform(validation_data[NUMERIC_COLUMNS])
scaled_test[NUMERIC_COLUMNS] = scaler.transform(test_data[NUMERIC_COLUMNS])
```

Standard scaling changes each value using this calculation:

```text
scaled value = (original value - training average) / training standard deviation
```

Therefore, the scaled training columns have an average close to 0 and a standard deviation close to 1. Validation and test columns may have different averages because they are measured against the training period, which is correct.

## 4. Leave binary columns unchanged

The script does not scale these indicator columns:

```text
is_rain_day
holiday_type_national
holiday_type_religious
is_lockdown
```

Their original 0/1 values are copied unchanged into each output file. The `timestamp` column is also unchanged.

## Why training-only scaling matters

If the scaler were fitted on all rows, it would use future validation and test information while preparing the training data. This is called data leakage. It can make model evaluation look better than it would be in real use.

Fitting on training data only and reusing that scaler everywhere prevents data leakage and gives a fair evaluation of the model.

## Run the script

From the project root, after running the cleaning script:

```powershell
.\myenv\Scripts\python.exe split_and_scale_smartgrid_data.py
```
