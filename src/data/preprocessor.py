"""Data preprocessing module: scaling and differencing for Branch O and Branch D representations."""

from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class Preprocessor:
    """
    Preprocessor for dual-branch time series preprocessing.
    
    Supports two representations:
    - Branch O (Original): standardized original values
    - Branch D (Differenced): standardized differences (Δx[t] = x[t] - x[t-1])
    
    Ensures scaler fitting on training data only, with separate scaler parameters
    for each branch for reproducibility and round-trip accuracy.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        numeric_cols: list
    ) -> None:
        """
        Initialize Preprocessor with DataFrame and column specifications.
        
        Args:
            df: Input DataFrame with time series data
            timestamp_col: Name of timestamp column
            numeric_cols: List of numeric column names to preprocess
            
        Raises:
            ValueError: If any specified column is missing from DataFrame
            ValueError: If numeric_cols is empty
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
        if not numeric_cols:
            raise ValueError("numeric_cols cannot be empty")
        
        missing_cols = [col for col in numeric_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")
        
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.numeric_cols = numeric_cols
        
        # Scaler storage: {branch_name: StandardScaler instance}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Scaler parameters storage for reproducibility: {branch_name: {'mean': ..., 'scale': ...}}
        self.scaler_params: Dict[str, Dict[str, np.ndarray]] = {}
        
        # Store last values for Branch D integration during inverse transform
        self.last_values: Optional[np.ndarray] = None
    
    def fit_scalers(self, df_train: pd.DataFrame) -> Dict[str, StandardScaler]:
        """
        Fit StandardScaler on training data for both branches.
        
        Computes mean and std on training data only, then stores scaler parameters
        for reproducibility and round-trip accuracy.
        
        Args:
            df_train: Training DataFrame (typically 2020-2023-01-01)
            
        Returns:
            dict: Fitted scalers with keys 'branch_o' and 'branch_d'
            
        Raises:
            ValueError: If df_train is empty or has missing values in numeric_cols
        """
        if len(df_train) == 0:
            raise ValueError("Training DataFrame is empty")
        
        # Check for missing values
        train_data = df_train[self.numeric_cols]
        if train_data.isnull().any().any():
            raise ValueError("Training data contains missing values in numeric columns")
        
        # Extract training data as numpy array for fitting
        train_values = train_data.values  # Shape: (n_train, n_channels)
        
        # Fit scaler for Branch O (original representation)
        scaler_o = StandardScaler()
        scaler_o.fit(train_values)
        self.scalers['branch_o'] = scaler_o
        self.scaler_params['branch_o'] = {
            'mean': scaler_o.mean_.copy(),
            'scale': scaler_o.scale_.copy()
        }
        
        # For Branch D, fit scaler on differences computed from training data
        # Compute differences: Δx[t] = x[t] - x[t-1]
        diff_values = np.diff(train_values, axis=0)  # Shape: (n_train - 1, n_channels)
        
        if len(diff_values) == 0:
            raise ValueError("Not enough training data to compute differences (need at least 2 rows)")
        
        # Fit scaler for Branch D (differenced representation)
        scaler_d = StandardScaler()
        scaler_d.fit(diff_values)
        self.scalers['branch_d'] = scaler_d
        self.scaler_params['branch_d'] = {
            'mean': scaler_d.mean_.copy(),
            'scale': scaler_d.scale_.copy()
        }
        
        return self.scalers
    
    def forward_branch_o(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply standardization to Branch O (original representation).
        
        Standardizes the original values using the fitted scaler.
        Formula: z = (x - mean) / scale
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            DataFrame: Scaled values with same index and column names as input
            
        Raises:
            RuntimeError: If scalers not yet fitted via fit_scalers()
            ValueError: If df has missing values in numeric_cols
        """
        if 'branch_o' not in self.scalers:
            raise RuntimeError("Scalers not fitted. Call fit_scalers() first.")
        
        if df[self.numeric_cols].isnull().any().any():
            raise ValueError("Input data contains missing values in numeric columns")
        
        # Extract numeric columns
        data = df[self.numeric_cols].values  # Shape: (n_rows, n_channels)
        
        # Apply scaler transformation
        scaled = self.scalers['branch_o'].transform(data)
        
        # Return as DataFrame with original index and columns
        return pd.DataFrame(
            scaled,
            index=df.index,
            columns=self.numeric_cols
        )
    
    def forward_branch_d(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply differencing and standardization to Branch D (differenced representation).
        
        Computes first differences: Δx[t] = x[t] - x[t-1]
        Then standardizes differences using fitted scaler.
        Preserves first original value for reconstruction during inverse transform.
        
        Args:
            df: DataFrame to preprocess
            
        Returns:
            tuple: (scaled_differences_df, first_level)
                - scaled_differences_df: Differenced+scaled DataFrame (n_rows - 1, n_channels)
                - first_level: First row of original values (1D array, shape (n_channels,))
                
        Raises:
            RuntimeError: If scalers not yet fitted via fit_scalers()
            ValueError: If df has fewer than 2 rows or has missing values
        """
        if 'branch_d' not in self.scalers:
            raise RuntimeError("Scalers not fitted. Call fit_scalers() first.")
        
        if len(df) < 2:
            raise ValueError("Input DataFrame must have at least 2 rows for differencing")
        
        if df[self.numeric_cols].isnull().any().any():
            raise ValueError("Input data contains missing values in numeric columns")
        
        # Extract numeric columns
        data = df[self.numeric_cols].values  # Shape: (n_rows, n_channels)
        
        # Preserve first values for integration during inverse transform
        first_level = data[0, :].copy()  # Shape: (n_channels,)
        
        # Compute first differences: Δx[t] = x[t] - x[t-1]
        diff_values = np.diff(data, axis=0)  # Shape: (n_rows - 1, n_channels)
        
        # Apply scaler transformation to differences
        scaled_diff = self.scalers['branch_d'].transform(diff_values)
        
        # Return as DataFrame with adjusted index and columns
        # Index: drop first row (since we have one fewer row after differencing)
        diff_df = pd.DataFrame(
            scaled_diff,
            index=df.index[1:],  # Skip first row since we differenced
            columns=self.numeric_cols
        )
        
        return diff_df, first_level
    
    def inverse_branch_o(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform: unscale Branch O from standardized to original units.
        
        Applies inverse standardization to recover original-scale values.
        Formula: x = z * scale + mean
        
        Args:
            df_scaled: Scaled DataFrame
            
        Returns:
            DataFrame: Original-scale values with same index and columns as input
            
        Raises:
            RuntimeError: If scalers not yet fitted
        """
        if 'branch_o' not in self.scalers:
            raise RuntimeError("Scalers not fitted. Call fit_scalers() first.")
        
        # Extract scaled data
        scaled = df_scaled[self.numeric_cols].values  # Shape: (n_rows, n_channels)
        
        # Apply inverse transformation
        original = self.scalers['branch_o'].inverse_transform(scaled)
        
        # Return as DataFrame with original index and columns
        return pd.DataFrame(
            original,
            index=df_scaled.index,
            columns=self.numeric_cols
        )
    
    def inverse_branch_d(
        self,
        df_scaled_diff: pd.DataFrame,
        first_level: np.ndarray
    ) -> pd.DataFrame:
        """
        Inverse transform: unscale and integrate Branch D to recover original units.
        
        1. Unscale differences from standardized to original units
        2. Integrate: x[t] = x[t-1] + Δx[t] (recursively reconstructs levels)
        3. Prepend first_level to first row to get complete sequence
        
        Args:
            df_scaled_diff: Differenced+scaled DataFrame (typically n - 1 rows)
                            Should have index starting from original_index[1]
            first_level: First level values (1D array, shape (n_channels,)) used as reference
            
        Returns:
            DataFrame: Original-scale level values, row count = len(df_scaled_diff) + 1
                       Index covers the same period as df_scaled_diff + one row before
            
        Raises:
            RuntimeError: If scalers not yet fitted
            ValueError: If first_level shape doesn't match n_channels
        """
        if 'branch_d' not in self.scalers:
            raise RuntimeError("Scalers not fitted. Call fit_scalers() first.")
        
        if len(first_level) != len(self.numeric_cols):
            raise ValueError(
                f"first_level shape {len(first_level)} doesn't match "
                f"n_channels {len(self.numeric_cols)}"
            )
        
        # Extract scaled differences
        scaled_diff = df_scaled_diff[self.numeric_cols].values  # Shape: (n_rows, n_channels)
        
        # Inverse transform differences: unscale to original units
        diff_original = self.scalers['branch_d'].inverse_transform(scaled_diff)
        
        # Integrate differences to recover levels: x[t] = x[t-1] + Δx[t]
        # Start with first_level as initial value
        n_rows = len(diff_original) + 1  # +1 because we have one fewer row after differencing
        integrated = np.zeros((n_rows, len(self.numeric_cols)), dtype=diff_original.dtype)
        
        # Set first row to first_level (reference point)
        integrated[0, :] = first_level
        
        # Recursively integrate differences
        for t in range(1, n_rows):
            integrated[t, :] = integrated[t - 1, :] + diff_original[t - 1, :]
        
        # Reconstruct original index with first row prepended
        # The df_scaled_diff.index starts at original_index[1], so we need to go back one
        if len(df_scaled_diff) > 0:
            first_diff_idx = df_scaled_diff.index[0]
            if isinstance(first_diff_idx, pd.Timestamp):
                # For datetime index, go back one period (assuming hourly)
                first_level_idx = first_diff_idx - pd.Timedelta(hours=1)
                new_index = pd.DatetimeIndex([first_level_idx] + df_scaled_diff.index.tolist())
            else:
                # For integer or other index, try to infer step
                try:
                    # Assume regular spacing
                    if len(df_scaled_diff.index) > 1:
                        step = df_scaled_diff.index[1] - df_scaled_diff.index[0]
                    else:
                        step = 1
                    first_level_idx = first_diff_idx - step
                except (TypeError, ValueError):
                    # If we can't compute step, just use -1
                    first_level_idx = first_diff_idx - 1
                
                new_index = pd.Index([first_level_idx] + df_scaled_diff.index.tolist())
        else:
            # If no data, create default index
            new_index = pd.RangeIndex(n_rows)
        
        return pd.DataFrame(
            integrated,
            index=new_index,
            columns=self.numeric_cols
        )
    
    def get_scalers(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Retrieve scaler parameters for reproducibility.
        
        Returns:
            dict: Scaler parameters with structure:
                {
                    'branch_o': {'mean': array, 'scale': array},
                    'branch_d': {'mean': array, 'scale': array}
                }
                
        Raises:
            RuntimeError: If scalers not yet fitted
        """
        if not self.scaler_params:
            raise RuntimeError("Scalers not fitted. Call fit_scalers() first.")
        
        return self.scaler_params
    
    def save_scalers(self, output_path: str) -> None:
        """
        Save scaler parameters to pickle files for reproducibility.
        
        Creates two pickle files:
        - {output_path}/scalers_branch_o.pkl
        - {output_path}/scalers_branch_d.pkl
        
        Args:
            output_path: Directory path where scalers will be saved
            
        Raises:
            RuntimeError: If scalers not yet fitted
            FileNotFoundError: If output directory doesn't exist
        """
        if not self.scaler_params:
            raise RuntimeError("Scalers not fitted. Call fit_scalers() first.")
        
        output_dir = Path(output_path)
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory {output_dir} does not exist")
        
        # Save Branch O scaler
        o_path = output_dir / 'scalers_branch_o.pkl'
        with open(o_path, 'wb') as f:
            pickle.dump(self.scaler_params['branch_o'], f)
        
        # Save Branch D scaler
        d_path = output_dir / 'scalers_branch_d.pkl'
        with open(d_path, 'wb') as f:
            pickle.dump(self.scaler_params['branch_d'], f)
    
    @staticmethod
    def load_scalers(output_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load scaler parameters from pickle files.
        
        Args:
            output_path: Directory path where scalers are saved
            
        Returns:
            dict: Scaler parameters {branch_name: {mean, scale}}
            
        Raises:
            FileNotFoundError: If pickle files not found
        """
        output_dir = Path(output_path)
        
        o_path = output_dir / 'scalers_branch_o.pkl'
        d_path = output_dir / 'scalers_branch_d.pkl'
        
        if not o_path.exists():
            raise FileNotFoundError(f"Branch O scaler not found at {o_path}")
        if not d_path.exists():
            raise FileNotFoundError(f"Branch D scaler not found at {d_path}")
        
        with open(o_path, 'rb') as f:
            branch_o = pickle.load(f)
        
        with open(d_path, 'rb') as f:
            branch_d = pickle.load(f)
        
        return {
            'branch_o': branch_o,
            'branch_d': branch_d
        }
