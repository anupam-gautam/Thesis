"""Data audit module for comprehensive dataset quality analysis."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from src.data.loader import DataLoader
from src.utils.checksums import md5_file


class DataAudit:
    """
    Analyzes dataset quality and generates comprehensive audit reports.
    
    Computes statistics including:
    - File metadata (path, MD5 checksum)
    - Structural info (row/column counts, dtypes)
    - Temporal info (timestamp range, frequency verification)
    - Data quality (duplicates, missing values, numeric ranges)
    - Target column statistics
    """
    
    def __init__(self, csv_path: str, loader: Optional[DataLoader] = None) -> None:
        """
        Initialize DataAudit.
        
        Args:
            csv_path: Path to the CSV file to audit
            loader: Optional pre-loaded DataLoader instance. If None, creates new instance.
            
        Raises:
            FileNotFoundError: If CSV file does not exist
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Use provided loader or create new one
        if loader is None:
            self.loader = DataLoader(str(self.csv_path))
            self.loader.load()
        else:
            self.loader = loader
        
        self.audit_dict: Dict[str, Any] = {}
    
    def run_audit(self) -> Dict[str, Any]:
        """
        Execute comprehensive audit and return all statistics.
        
        Computes:
        - File metadata (path, checksum, row/column counts)
        - Column information (dtypes, null counts)
        - Timestamp analysis (first, last, timezone if detectable, frequency)
        - Data quality (duplicates, missing values)
        - Numeric statistics (ranges, mean, std)
        - Target column statistics
        - Cleaning notes
        
        Returns:
            Dictionary containing all audit statistics
        """
        if self.loader.df is None:
            raise RuntimeError("DataLoader not initialized. Call load() first.")
        
        df = self.loader.df
        
        # 1. File metadata
        self.audit_dict['file_path'] = str(self.csv_path)
        self.audit_dict['md5_checksum'] = md5_file(self.csv_path)
        self.audit_dict['row_count'] = len(df)
        self.audit_dict['column_count'] = len(df.columns)
        self.audit_dict['audit_timestamp'] = datetime.now().isoformat()
        
        # 2. Column information
        self.audit_dict['columns'] = {}
        for col in df.columns:
            self.audit_dict['columns'][col] = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isna().sum()),
                'null_percentage': float(
                    (df[col].isna().sum() / len(df) * 100) if len(df) > 0 else 0
                )
            }
        
        # 3. Timestamp range and frequency analysis
        self._analyze_temporal_data()
        
        # 4. Duplicate timestamps
        self._analyze_duplicates()
        
        # 5. Missing values summary
        self._analyze_missing_values()
        
        # 6. Numeric column ranges
        self._analyze_numeric_ranges()
        
        # 7. Target column statistics
        if self.loader.target_col:
            self._analyze_target_column()
        
        # 8. Cleaning notes
        self.audit_dict['cleaning_notes'] = ''
        
        return self.audit_dict
    
    def _analyze_temporal_data(self) -> None:
        """Analyze timestamp column: range, frequency, and coverage."""
        if not self.loader.timestamp_col:
            self.audit_dict['timestamp_range'] = None
            self.audit_dict['frequency_check'] = None
            return
        
        df = self.loader.df
        ts_col = self.loader.timestamp_col
        
        # Parse timestamps if not already datetime
        ts_data = pd.to_datetime(df[ts_col], errors='coerce')
        ts_valid = ts_data.dropna()
        
        if len(ts_valid) == 0:
            self.audit_dict['timestamp_range'] = None
            self.audit_dict['frequency_check'] = None
            return
        
        # Timestamp range
        first_ts = ts_valid.min()
        last_ts = ts_valid.max()
        
        self.audit_dict['timestamp_range'] = {
            'first': first_ts.isoformat(),
            'last': last_ts.isoformat(),
            'timezone': str(first_ts.tz) if first_ts.tz else None
        }
        
        # Frequency analysis (hourly data expected)
        time_diff = (last_ts - first_ts).total_seconds() / 3600  # hours
        expected_hourly_rows = int(time_diff) + 1  # +1 for inclusive range
        actual_rows = len(ts_valid)
        
        # Calculate coverage percentage
        coverage_pct = (actual_rows / expected_hourly_rows * 100) if expected_hourly_rows > 0 else 0
        
        # Identify missing hours if any
        missing_hours = expected_hourly_rows - actual_rows
        
        self.audit_dict['frequency_check'] = {
            'expected_hourly_rows': expected_hourly_rows,
            'actual_rows': actual_rows,
            'coverage_percentage': round(coverage_pct, 2),
            'missing_hours': missing_hours,
            'has_full_coverage': missing_hours == 0
        }
    
    def _analyze_duplicates(self) -> None:
        """Analyze duplicate timestamps."""
        if not self.loader.timestamp_col:
            self.audit_dict['duplicate_timestamps'] = {
                'count': 0,
                'percentage': 0.0
            }
            return
        
        df = self.loader.df
        ts_col = self.loader.timestamp_col
        
        ts_data = pd.to_datetime(df[ts_col], errors='coerce')
        ts_valid = ts_data.dropna()
        
        duplicate_count = len(ts_valid) - len(ts_valid.drop_duplicates())
        duplicate_pct = (duplicate_count / len(ts_valid) * 100) if len(ts_valid) > 0 else 0
        
        self.audit_dict['duplicate_timestamps'] = {
            'count': int(duplicate_count),
            'percentage': round(duplicate_pct, 2)
        }
    
    def _analyze_missing_values(self) -> None:
        """Generate missing values summary."""
        df = self.loader.df
        
        missing_summary = {}
        for col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0
                missing_summary[col] = {
                    'count': int(null_count),
                    'percentage': round(null_pct, 2)
                }
        
        self.audit_dict['missing_values_summary'] = missing_summary
    
    def _analyze_numeric_ranges(self) -> None:
        """Analyze min, max, mean, std for numeric columns."""
        df = self.loader.df
        
        numeric_ranges = {}
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    numeric_ranges[col] = {
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'count': int(len(col_data))
                    }
        
        self.audit_dict['numeric_column_ranges'] = numeric_ranges
    
    def _analyze_target_column(self) -> None:
        """Analyze target column statistics."""
        if not self.loader.target_col:
            self.audit_dict['target_column_stats'] = None
            return
        
        df = self.loader.df
        target_col = self.loader.target_col
        
        target_data = df[target_col]
        
        self.audit_dict['target_column_stats'] = {
            'column_name': target_col,
            'min': float(target_data.min()),
            'max': float(target_data.max()),
            'mean': float(target_data.mean()),
            'std': float(target_data.std()),
            'null_count': int(target_data.isna().sum())
        }
    
    def get_audit_dict(self) -> Dict[str, Any]:
        """
        Return the complete audit dictionary.
        
        Returns:
            Dictionary with all audit statistics
            
        Raises:
            RuntimeError: If run_audit() has not been called yet
        """
        if not self.audit_dict:
            raise RuntimeError("Audit not yet run. Call run_audit() first.")
        
        return self.audit_dict
    
    def save_audit_json(self, output_path: str) -> None:
        """
        Save audit results to JSON file.
        
        Args:
            output_path: Path where to save the JSON file
            
        Raises:
            RuntimeError: If run_audit() has not been called yet
        """
        if not self.audit_dict:
            raise RuntimeError("Audit not yet run. Call run_audit() first.")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.audit_dict, f, indent=2)
        
        print(f"✓ Audit saved to: {output_file}")
    
    def print_audit_summary(self) -> None:
        """Print human-readable audit report to console."""
        if not self.audit_dict:
            raise RuntimeError("Audit not yet run. Call run_audit() first.")
        
        print("\n" + "=" * 80)
        print("DATA AUDIT REPORT")
        print("=" * 80)
        
        # File information
        print(f"\nFile: {self.audit_dict['file_path']}")
        print(f"MD5 Checksum: {self.audit_dict['md5_checksum']}")
        print(f"Audit Timestamp: {self.audit_dict['audit_timestamp']}")
        
        # Dimensions
        print(f"\nDimensions:")
        print(f"  Rows: {self.audit_dict['row_count']:,}")
        print(f"  Columns: {self.audit_dict['column_count']}")
        
        # Timestamp range
        ts_range = self.audit_dict.get('timestamp_range')
        if ts_range:
            print(f"\nTimestamp Range:")
            print(f"  First: {ts_range['first']}")
            print(f"  Last:  {ts_range['last']}")
            if ts_range['timezone']:
                print(f"  Timezone: {ts_range['timezone']}")
        
        # Frequency check
        freq_check = self.audit_dict.get('frequency_check')
        if freq_check:
            print(f"\nFrequency Check (Hourly):")
            print(f"  Expected rows: {freq_check['expected_hourly_rows']:,}")
            print(f"  Actual rows: {freq_check['actual_rows']:,}")
            print(f"  Coverage: {freq_check['coverage_percentage']:.2f}%")
            if freq_check['missing_hours'] > 0:
                print(f"  Missing hours: {freq_check['missing_hours']}")
            else:
                print(f"  Missing hours: 0 (complete coverage)")
        
        # Duplicate timestamps
        dup_ts = self.audit_dict.get('duplicate_timestamps')
        if dup_ts:
            print(f"\nDuplicate Timestamps:")
            print(f"  Count: {dup_ts['count']}")
            print(f"  Percentage: {dup_ts['percentage']:.2f}%")
        
        # Missing values
        missing = self.audit_dict.get('missing_values_summary')
        if missing:
            print(f"\nMissing Values:")
            for col, stats in missing.items():
                print(f"  {col:30s}: {stats['count']:6d} ({stats['percentage']:6.2f}%)")
        else:
            print(f"\nMissing Values: None")
        
        # Columns info
        print(f"\nColumns:")
        for col, meta in self.audit_dict['columns'].items():
            null_str = f"null: {meta['null_count']}" if meta['null_count'] > 0 else "no nulls"
            print(f"  {col:30s} | dtype: {meta['dtype']:10s} | {null_str}")
        
        # Numeric column ranges
        numeric_ranges = self.audit_dict.get('numeric_column_ranges')
        if numeric_ranges:
            print(f"\nNumeric Column Statistics:")
            for col, stats in numeric_ranges.items():
                print(f"\n  {col}:")
                print(f"    Min:  {stats['min']:15.6f}")
                print(f"    Max:  {stats['max']:15.6f}")
                print(f"    Mean: {stats['mean']:15.6f}")
                print(f"    Std:  {stats['std']:15.6f}")
        
        # Target column stats
        target_stats = self.audit_dict.get('target_column_stats')
        if target_stats:
            print(f"\nTarget Column: {target_stats['column_name']}")
            print(f"  Min:       {target_stats['min']:15.6f}")
            print(f"  Max:       {target_stats['max']:15.6f}")
            print(f"  Mean:      {target_stats['mean']:15.6f}")
            print(f"  Std:       {target_stats['std']:15.6f}")
            print(f"  Null count: {target_stats['null_count']}")
        
        # Cleaning notes
        if self.audit_dict.get('cleaning_notes'):
            print(f"\nCleaning Notes:")
            print(f"  {self.audit_dict['cleaning_notes']}")
        
        print("\n" + "=" * 80 + "\n")
