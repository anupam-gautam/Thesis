"""Chronological time series data splitting module."""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
from datetime import datetime


class DataSplitter:
    """
    Creates chronological train/validation/test splits for time series data.
    
    Ensures no temporal leakage by enforcing strict chronological boundaries.
    Validates split integrity and data continuity.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        target_col: str,
        train_end: str = '2023-01-01',
        val_end: str = '2024-01-01'
    ) -> None:
        """
        Initialize DataSplitter with data and split boundaries.
        
        Args:
            df: Input DataFrame with time series data
            timestamp_col: Name of timestamp column
            target_col: Name of target column
            train_end: End date for training split (default: '2023-01-01')
            val_end: End date for validation split (default: '2024-01-01')
            
        Raises:
            ValueError: If timestamp_col or target_col not in DataFrame
            ValueError: If boundaries are invalid (train_end >= val_end)
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        
        # Convert timestamp column to datetime if not already
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        
        # Parse boundary dates
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)
        
        # Validate boundaries
        if self.train_end >= self.val_end:
            raise ValueError(
                f"Invalid boundaries: train_end ({self.train_end}) must be "
                f"strictly before val_end ({self.val_end})"
            )
        
        # Get data boundaries
        self.data_start = self.df[self.timestamp_col].min()
        self.data_end = self.df[self.timestamp_col].max()
        
        # Initialize split results
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self._split_info: Optional[Dict[str, Any]] = None
    
    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/validation/test splits.
        
        Splits are created using strict temporal boundaries:
        - Training: timestamp < train_end
        - Validation: train_end <= timestamp < val_end
        - Test: timestamp >= val_end
        
        Returns:
            tuple: (train_df, val_df, test_df) - DataFrames for each split
            
        Raises:
            ValueError: If any split is empty
            ValueError: If data is lost during splitting (row count mismatch)
            ValueError: If boundaries overlap or are invalid
        """
        # Create splits using strict chronological boundaries
        self.train_df = self.df[self.df[self.timestamp_col] < self.train_end].copy()
        self.val_df = self.df[
            (self.df[self.timestamp_col] >= self.train_end) &
            (self.df[self.timestamp_col] < self.val_end)
        ].copy()
        self.test_df = self.df[self.df[self.timestamp_col] >= self.val_end].copy()
        
        # Verify no empty splits
        if len(self.train_df) == 0:
            raise ValueError(
                f"Training split is empty. No data before {self.train_end}. "
                f"Data range: {self.data_start} to {self.data_end}"
            )
        if len(self.val_df) == 0:
            raise ValueError(
                f"Validation split is empty. No data between "
                f"{self.train_end} and {self.val_end}. "
                f"Data range: {self.data_start} to {self.data_end}"
            )
        if len(self.test_df) == 0:
            raise ValueError(
                f"Test split is empty. No data after {self.val_end}. "
                f"Data range: {self.data_start} to {self.data_end}"
            )
        
        # Verify no data loss
        total_split_rows = len(self.train_df) + len(self.val_df) + len(self.test_df)
        if total_split_rows != len(self.df):
            raise ValueError(
                f"Data loss during splitting. Original rows: {len(self.df)}, "
                f"split rows: {total_split_rows}"
            )
        
        # Verify chronological order (no overlap)
        train_max = self.train_df[self.timestamp_col].max()
        val_min = self.val_df[self.timestamp_col].min()
        val_max = self.val_df[self.timestamp_col].max()
        test_min = self.test_df[self.timestamp_col].min()
        
        if train_max >= val_min:
            raise ValueError(
                f"Train/val boundary overlap: train max {train_max} >= val min {val_min}"
            )
        if val_max >= test_min:
            raise ValueError(
                f"Val/test boundary overlap: val max {val_max} >= test min {test_min}"
            )
        
        # Cache split info
        self._split_info = None  # Reset to force recalculation
        
        return self.train_df, self.val_df, self.test_df
    
    def get_split_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the splits.
        
        Returns:
            dict with keys:
                - train: {start_date, end_date, row_count}
                - validation: {start_date, end_date, row_count}
                - test: {start_date, end_date, row_count}
                - total_rows: Total row count across all splits
                
        Raises:
            RuntimeError: If split() has not been called yet
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise RuntimeError(
                "Splits not created yet. Call split() method first."
            )
        
        # Build split info dictionary
        info = {
            'train': {
                'start_date': str(self.train_df[self.timestamp_col].min()),
                'end_date': str(self.train_df[self.timestamp_col].max()),
                'row_count': len(self.train_df)
            },
            'validation': {
                'start_date': str(self.val_df[self.timestamp_col].min()),
                'end_date': str(self.val_df[self.timestamp_col].max()),
                'row_count': len(self.val_df)
            },
            'test': {
                'start_date': str(self.test_df[self.timestamp_col].min()),
                'end_date': str(self.test_df[self.timestamp_col].max()),
                'row_count': len(self.test_df)
            },
            'total_rows': len(self.train_df) + len(self.val_df) + len(self.test_df)
        }
        
        return info
    
    def print_split_info(self) -> None:
        """
        Print human-readable split summary to console.
        
        Displays:
        - Split boundaries
        - Row counts and percentages
        - Date ranges for each split
        - Data integrity verification
        
        Raises:
            RuntimeError: If split() has not been called yet
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise RuntimeError(
                "Splits not created yet. Call split() method first."
            )
        
        info = self.get_split_info()
        total = info['total_rows']
        
        print("\n" + "=" * 80)
        print("CHRONOLOGICAL SPLIT SUMMARY")
        print("=" * 80)
        
        print(f"\nSplit Boundaries:")
        print(f"  Training:   < {self.train_end.date()}")
        print(f"  Validation: {self.train_end.date()} to {self.val_end.date()}")
        print(f"  Test:       >= {self.val_end.date()}")
        
        print(f"\nTraining Split:")
        print(f"  Date range: {info['train']['start_date']} to {info['train']['end_date']}")
        print(f"  Rows: {info['train']['row_count']:,} ({info['train']['row_count']/total*100:.1f}%)")
        
        print(f"\nValidation Split:")
        print(f"  Date range: {info['validation']['start_date']} to {info['validation']['end_date']}")
        print(f"  Rows: {info['validation']['row_count']:,} ({info['validation']['row_count']/total*100:.1f}%)")
        
        print(f"\nTest Split:")
        print(f"  Date range: {info['test']['start_date']} to {info['test']['end_date']}")
        print(f"  Rows: {info['test']['row_count']:,} ({info['test']['row_count']/total*100:.1f}%)")
        
        print(f"\nTotal Rows: {total:,}")
        print(f"  (train + val + test = {info['train']['row_count']} + "
              f"{info['validation']['row_count']} + "
              f"{info['test']['row_count']})")
        
        print("\n✓ Data integrity verified:")
        print(f"  - No overlapping boundaries")
        print(f"  - No data loss (all {total:,} rows accounted for)")
        print(f"  - Chronological order preserved (no shuffling)")
        print(f"  - No temporal leakage between splits")
        
        print("=" * 80 + "\n")
