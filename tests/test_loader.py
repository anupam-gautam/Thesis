"""Unit tests for data loader."""

import tempfile
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path as PathlibPath

# Add parent directory to path for imports
sys.path.insert(0, str(PathlibPath(__file__).parent.parent))

from src.data.loader import DataLoader


def test_loader_valid_csv():
    """Test loading valid CSV with correct schema."""
    # Create temp CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_content = """timestamp,temperature,is_rain_day,holiday_type_national,holiday_type_religious,is_lockdown,load_kw
2020-01-01 00:00,25.2,0,1,0,0,36521.78538
2020-01-01 01:00,25.2,0,1,0,0,34934.33925
2020-01-01 02:00,25.2,0,1,0,0,36899.22135"""
        f.write(csv_content)
        temp_path = f.name
    
    try:
        # Load and validate
        loader = DataLoader(temp_path)
        df = loader.load()
        
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"
        assert loader.timestamp_col == 'timestamp', f"Expected 'timestamp', got {loader.timestamp_col}"
        assert loader.target_col == 'load_kw', f"Expected 'load_kw', got {loader.target_col}"
        assert pd.api.types.is_datetime64_any_dtype(df[loader.timestamp_col]), \
            f"Expected datetime type, got {df[loader.timestamp_col].dtype}"
        
        print("✓ test_loader_valid_csv PASSED")
    finally:
        Path(temp_path).unlink()


def test_loader_missing_file():
    """Test error on missing file."""
    try:
        loader = DataLoader('/nonexistent/path/data.csv')
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError as e:
        print(f"✓ test_loader_missing_file PASSED (correctly raised: {type(e).__name__})")


def test_schema_summary():
    """Test summary generation."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_content = """timestamp,temperature,is_rain_day,holiday_type_national,holiday_type_religious,is_lockdown,load_kw
2020-01-01 00:00,25.2,0,1,0,0,36521.78
2020-01-01 01:00,25.3,0,1,0,0,34934.34"""
        f.write(csv_content)
        temp_path = f.name
    
    try:
        loader = DataLoader(temp_path)
        df = loader.load()
        summary = loader.get_summary()
        
        assert summary['row_count'] == 2, f"Expected 2 rows, got {summary['row_count']}"
        assert summary['column_count'] == 7, f"Expected 7 columns, got {summary['column_count']}"
        assert summary['target_column'] == 'load_kw', f"Expected 'load_kw', got {summary['target_column']}"
        assert summary['timestamp_column'] == 'timestamp', f"Expected 'timestamp', got {summary['timestamp_column']}"
        
        # Verify structure
        assert 'columns' in summary, "Missing 'columns' key in summary"
        assert 'timestamp_range' in summary, "Missing 'timestamp_range' key in summary"
        assert summary['timestamp_range'] is not None, "timestamp_range should not be None"
        
        print("✓ test_schema_summary PASSED")
    finally:
        Path(temp_path).unlink()


def test_loader_on_actual_data():
    """Test loading actual CLEAN_smartgrid CSV."""
    csv_path = 'd:\\Masters\\Final Semester\\Thesis\\Coding\\DataSet\\CLEAN_smartgrid_5yr_hourly_natural.csv'
    
    if not Path(csv_path).exists():
        print(f"⊘ test_loader_on_actual_data SKIPPED (data file not found: {csv_path})")
        return
    
    try:
        loader = DataLoader(csv_path)
        df = loader.load()
        
        # Verify basic structure
        assert len(df) > 0, "DataFrame is empty"
        assert loader.timestamp_col == 'timestamp', f"Expected 'timestamp', got {loader.timestamp_col}"
        assert loader.target_col == 'load_kw', f"Expected 'load_kw', got {loader.target_col}"
        
        # Verify required columns
        required_cols = ['timestamp', 'load_kw', 'temperature', 'is_rain_day', 
                        'holiday_type_national', 'holiday_type_religious', 'is_lockdown']
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Verify timestamp is datetime
        assert df[loader.timestamp_col].dtype == 'datetime64[ns]', \
            f"Timestamp should be datetime64[ns], got {df[loader.timestamp_col].dtype}"
        
        # Verify target is numeric
        assert pd.api.types.is_numeric_dtype(df[loader.target_col]), \
            f"Target column {loader.target_col} should be numeric"
        
        # Print actual data summary
        print(f"\n✓ test_loader_on_actual_data PASSED")
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Load range: {df['load_kw'].min():.2f} to {df['load_kw'].max():.2f} kW")
        
    except Exception as e:
        print(f"✗ test_loader_on_actual_data FAILED: {e}")
        raise


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running unit tests for DataLoader")
    print("="*60 + "\n")
    
    test_loader_valid_csv()
    test_loader_missing_file()
    test_schema_summary()
    test_loader_on_actual_data()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")
