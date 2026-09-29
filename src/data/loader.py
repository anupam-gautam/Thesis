"""Data loader for time series data with auto-detection and schema validation."""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import warnings


class DataLoader:
    """
    Loads and validates CSV time series data with automatic column detection.
    
    Features:
    - Auto-detects timestamp column by attempting datetime parsing
    - Auto-detects target column by name matching
    - Validates schema and converts data types
    - Provides summary statistics and metadata
    """
    
    # Target column name patterns to search for (in order of preference)
    TARGET_PATTERNS = ['load_kw', 'load', 'consumption', 'demand', 'power']
    
    def __init__(self, csv_path: str) -> None:
        """
        Initialize DataLoader with a CSV file path.
        
        Args:
            csv_path: Path to the CSV file
            
        Raises:
            FileNotFoundError: If the CSV file does not exist
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df: Optional[pd.DataFrame] = None
        self.timestamp_col: Optional[str] = None
        self.target_col: Optional[str] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load and validate CSV data with auto-detection of key columns.
        
        Process:
        1. Read CSV using pandas
        2. Auto-detect timestamp column
        3. Auto-detect target column
        4. Validate schema
        5. Convert data types
        
        Returns:
            pandas.DataFrame: Loaded and validated data
            
        Raises:
            ValueError: If timestamp or target column cannot be detected
        """
        # Read CSV file
        self.df = pd.read_csv(self.csv_path)
        
        # Detect and set timestamp column
        self._detect_timestamp_column()
        
        # Detect and set target column
        self._detect_target_column()
        
        # Validate schema and convert types
        self._validate_schema()
        
        return self.df
    
    def _detect_timestamp_column(self) -> None:
        """
        Auto-detect timestamp column by attempting to parse each column as datetime.
        
        Iterates through columns and tries pd.to_datetime() on each.
        Sets self.timestamp_col to the first column that can be parsed as datetime.
        Prints detection message upon success.
        
        Raises:
            ValueError: If no column can be parsed as datetime
        """
        if self.df is None:
            raise RuntimeError("DataFrame not loaded. Call load() first.")
        
        for col in self.df.columns:
            try:
                # Attempt to parse column as datetime
                pd.to_datetime(self.df[col], errors='coerce')
                # Check if at least some values were successfully parsed
                parsed = pd.to_datetime(self.df[col], errors='coerce')
                if parsed.notna().sum() > 0:
                    self.timestamp_col = col
                    print(f"✓ Detected timestamp column: '{col}'")
                    return
            except (ValueError, TypeError):
                continue
        
        raise ValueError(
            f"Could not detect timestamp column. Tried all columns: {list(self.df.columns)}"
        )
    
    def _detect_target_column(self) -> None:
        """
        Auto-detect target column by matching against known patterns.
        
        Searches for column names matching TARGET_PATTERNS in order of preference.
        Case-insensitive matching.
        Sets self.target_col upon finding a match.
        Prints detection message upon success.
        
        Raises:
            ValueError: If no target column pattern is found
        """
        if self.df is None:
            raise RuntimeError("DataFrame not loaded. Call load() first.")
        
        # Convert column names to lowercase for case-insensitive matching
        columns_lower = {col.lower(): col for col in self.df.columns}
        
        # Search for target patterns in order of preference
        for pattern in self.TARGET_PATTERNS:
            if pattern.lower() in columns_lower:
                self.target_col = columns_lower[pattern.lower()]
                print(f"✓ Detected target column: '{self.target_col}'")
                return
        
        raise ValueError(
            f"Could not detect target column. Expected one of {self.TARGET_PATTERNS}. "
            f"Available columns: {list(self.df.columns)}"
        )
    
    def _validate_schema(self) -> None:
        """
        Validate schema and convert data types.
        
        Validates:
        - Required columns (timestamp and target) are present
        - Timestamps are parsed to datetime64
        - Numeric columns are converted to float64 or int64
        
        Prints validation summary.
        
        Raises:
            ValueError: If required columns are missing
        """
        if self.df is None:
            raise RuntimeError("DataFrame not loaded. Call load() first.")
        
        if self.timestamp_col is None or self.target_col is None:
            raise ValueError("Timestamp and target columns must be detected first")
        
        # Validate required columns exist
        if self.timestamp_col not in self.df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_col}' not found in data")
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        
        # Parse timestamp column to datetime
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.df[self.timestamp_col] = pd.to_datetime(
                self.df[self.timestamp_col],
                errors='coerce'
            )
        
        # Convert numeric columns to appropriate types
        for col in self.df.columns:
            if col == self.timestamp_col:
                continue
            
            # Try to convert to numeric
            try:
                # Attempt numeric conversion
                numeric_data = pd.to_numeric(self.df[col], errors='coerce')
                
                # Check if conversion is meaningful (not all NaN)
                if numeric_data.notna().sum() > 0:
                    # Determine if integer or float
                    if (numeric_data == numeric_data.astype('Int64', errors='ignore')).all():
                        self.df[col] = numeric_data.astype('int64')
                    else:
                        self.df[col] = numeric_data.astype('float64')
            except (ValueError, TypeError):
                # Keep original type if conversion fails
                pass
        
        # Print validation summary
        num_rows = len(self.df)
        num_cols = len(self.df.columns)
        print(f"✓ Schema valid: {num_cols} columns, {num_rows} rows")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary as a dictionary.
        
        Returns:
            dict with keys:
                - file_path: Path to the CSV file
                - row_count: Number of rows
                - column_count: Number of columns
                - columns: Dict mapping column names to metadata
                    - dtype: Data type
                    - non_null_count: Number of non-null values
                    - null_count: Number of null values
                - timestamp_range: Dict with 'first' and 'last' timestamps
                - target_column: Name of target column
                - timestamp_column: Name of timestamp column
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        # Build column metadata
        columns_meta = {}
        for col in self.df.columns:
            columns_meta[col] = {
                'dtype': str(self.df[col].dtype),
                'non_null_count': int(self.df[col].notna().sum()),
                'null_count': int(self.df[col].isna().sum())
            }
        
        # Get timestamp range if timestamp column is detected
        timestamp_range = None
        if self.timestamp_col and self.timestamp_col in self.df.columns:
            ts_data = self.df[self.timestamp_col].dropna()
            if len(ts_data) > 0:
                timestamp_range = {
                    'first': str(ts_data.min()),
                    'last': str(ts_data.max())
                }
        
        return {
            'file_path': str(self.csv_path),
            'row_count': len(self.df),
            'column_count': len(self.df.columns),
            'columns': columns_meta,
            'timestamp_range': timestamp_range,
            'target_column': self.target_col,
            'timestamp_column': self.timestamp_col
        }
    
    def print_summary(self) -> None:
        """
        Print human-readable data summary to console.
        
        Displays:
        - File path
        - Number of rows and columns
        - Column details (dtype, null counts)
        - Timestamp range
        - Target column range (min/max/mean)
        - Overall null count summary
        """
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("DATA SUMMARY")
        print("=" * 70)
        
        # File and dimensions
        print(f"\nFile: {summary['file_path']}")
        print(f"Dimensions: {summary['row_count']} rows × {summary['column_count']} columns")
        
        # Timestamp range
        if summary['timestamp_range']:
            print(f"\nTimestamp range:")
            print(f"  First: {summary['timestamp_range']['first']}")
            print(f"  Last:  {summary['timestamp_range']['last']}")
        
        # Target column stats
        if self.target_col and self.target_col in self.df.columns:
            target_data = self.df[self.target_col].dropna()
            print(f"\nTarget column: {self.target_col}")
            print(f"  Min:   {target_data.min():.2f}")
            print(f"  Max:   {target_data.max():.2f}")
            print(f"  Mean:  {target_data.mean():.2f}")
            print(f"  Std:   {target_data.std():.2f}")
        
        # Column details
        print(f"\nColumns:")
        for col, meta in summary['columns'].items():
            null_pct = (meta['null_count'] / summary['row_count'] * 100) if summary['row_count'] > 0 else 0
            print(f"  {col:30s} | dtype: {meta['dtype']:10s} | "
                  f"non-null: {meta['non_null_count']:6d} | "
                  f"null: {meta['null_count']:6d} ({null_pct:5.2f}%)")
        
        print("=" * 70 + "\n")
