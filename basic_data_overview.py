import pandas as pd

# Load the dataset
csv_path = r"DataSet\RAW_smartgrid_5yr_hourly_natural.csv"
df = pd.read_csv(csv_path, parse_dates=['timestamp'])

print('=== Basic Data Characteristics ===')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('\nData types:')
print(df.dtypes)
print('\nMissing values:')
print(df.isnull().sum())
print('\nTime range:')
print('Start:', df['timestamp'].min())
print('End:', df['timestamp'].max())
print('\nBasic statistics for numeric columns:')
print(df.describe().round(2))
