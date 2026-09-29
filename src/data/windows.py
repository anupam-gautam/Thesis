"""Window generation and eligible origin manifest creation for time series forecasting."""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd


class WindowGenerator:
    """
    Generates eligible origins for time series windows with proper constraint enforcement.
    
    Ensures no target leakage by enforcing:
    - origin + lookback ≤ split boundary (enough history within split)
    - origin + lookback + horizon ≤ end-of-split (all targets within split)
    
    Stride = 1 (every eligible timestamp is considered an origin).
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        lookback_max: int,
        horizon_max: int
    ) -> None:
        """
        Initialize WindowGenerator with data and max constraint values.
        
        Args:
            df: DataFrame containing full time series data
            timestamp_col: Name of timestamp column
            lookback_max: Maximum lookback window length (in time steps)
            horizon_max: Maximum forecast horizon (in time steps)
            
        Raises:
            ValueError: If timestamp_col not in DataFrame
            ValueError: If lookback_max or horizon_max <= 0
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
        
        if lookback_max <= 0:
            raise ValueError(f"lookback_max must be > 0, got {lookback_max}")
        
        if horizon_max <= 0:
            raise ValueError(f"horizon_max must be > 0, got {horizon_max}")
        
        self.df = df.copy()
        self.timestamp_col = timestamp_col
        self.lookback_max = lookback_max
        self.horizon_max = horizon_max
        
        # Convert timestamp to datetime
        self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col])
        
        # Sort by timestamp to ensure chronological order
        self.df = self.df.sort_values(self.timestamp_col).reset_index(drop=True)
    
    def generate_origins(
        self,
        split_name: str,
        split_df: pd.DataFrame,
        lookback: int,
        horizon: int
    ) -> List[int]:
        """
        Generate eligible origin indices for a given split and L/H parameters.
        
        Constraint enforcement:
        - origin >= 0 (valid index in split_df)
        - origin + lookback <= len(split_df) (enough history within split)
        - origin + lookback + horizon <= len(split_df) (all targets within split)
        
        Stride = 1 (every valid index is considered).
        
        Args:
            split_name: Name of the split (e.g., 'train', 'val', 'test')
            split_df: DataFrame for this split
            lookback: Lookback window length for this configuration
            horizon: Forecast horizon for this configuration
            
        Returns:
            List of eligible origin indices (relative to split_df)
            
        Raises:
            ValueError: If lookback > lookback_max or horizon > horizon_max
            ValueError: If split is too small for any valid origins
        """
        if lookback > self.lookback_max:
            raise ValueError(
                f"lookback ({lookback}) exceeds max ({self.lookback_max}) "
                f"for split '{split_name}'"
            )
        
        if horizon > self.horizon_max:
            raise ValueError(
                f"horizon ({horizon}) exceeds max ({self.horizon_max}) "
                f"for split '{split_name}'"
            )
        
        split_len = len(split_df)
        eligible_origins = []
        
        # Maximum valid origin index: last index where origin + lookback + horizon <= split_len
        max_origin_idx = split_len - lookback - horizon
        
        # Generate all eligible origins (stride=1)
        for origin_idx in range(max_origin_idx + 1):
            # Verify constraints
            if origin_idx + lookback <= split_len and origin_idx + lookback + horizon <= split_len:
                eligible_origins.append(origin_idx)
        
        if len(eligible_origins) == 0:
            raise ValueError(
                f"No eligible origins for split '{split_name}': "
                f"split_len={split_len}, lookback={lookback}, horizon={horizon}. "
                f"Minimum required split length: {lookback + horizon}"
            )
        
        return eligible_origins
    
    def get_origin_manifest(
        self,
        experiments_config: Dict[str, Any],
        splits_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Generate origin manifests for all experiments and splits.
        
        Creates nested structure with eligible origins per experiment/split.
        
        Args:
            experiments_config: Dict of experiment configurations
                Expected keys: exp1_horizon, exp2_lookback, exp3a_representation, exp3b_temporal_shift
                Each experiment config should contain:
                  - lookback (or lookbacks list for exp2)
                  - horizon (or horizons list for exp1)
                  - representations (optional, for exp3a)
            splits_dict: Dict mapping split names to DataFrames
                Expected keys: 'train', 'validation', 'test'
        
        Returns:
            Nested dict structure:
            {
                'exp1_horizon': {
                    'train': {'origins': [0, 1, ..., N], 'count': N, 'lookback': 336, 'horizon': 96},
                    'validation': {...},
                    'test': {...}
                },
                'exp2_lookback': {
                    'train': {...},
                    ...
                },
                ...
            }
            
        Raises:
            KeyError: If required experiment configs or splits missing
            ValueError: If no eligible origins found for any configuration
        """
        manifest = {}
        
        # Process Experiment 1: Horizon Sweep (L=336 fixed)
        if 'exp1_horizon' in experiments_config:
            exp1_cfg = experiments_config['exp1_horizon']
            lookback = exp1_cfg.get('lookback', 336)
            horizons = exp1_cfg.get('horizons', [96, 192, 336, 720])
            
            manifest['exp1_horizon'] = {}
            
            for split_name, split_df in splits_dict.items():
                manifest['exp1_horizon'][split_name] = {
                    'horizons': []
                }
                
                # Generate origins for each horizon in exp1
                for horizon in horizons:
                    origins = self.generate_origins(split_name, split_df, lookback, horizon)
                    manifest['exp1_horizon'][split_name]['horizons'].append({
                        'horizon': horizon,
                        'origins': origins,
                        'count': len(origins)
                    })
        
        # Process Experiment 2: Lookback Sweep (H=192 fixed)
        if 'exp2_lookback' in experiments_config:
            exp2_cfg = experiments_config['exp2_lookback']
            horizon = exp2_cfg.get('horizon', 192)
            lookbacks = exp2_cfg.get('lookbacks', [96, 192, 336, 720])
            
            manifest['exp2_lookback'] = {}
            
            for split_name, split_df in splits_dict.items():
                manifest['exp2_lookback'][split_name] = {
                    'lookbacks': []
                }
                
                # Generate origins for each lookback in exp2
                for lookback in lookbacks:
                    origins = self.generate_origins(split_name, split_df, lookback, horizon)
                    manifest['exp2_lookback'][split_name]['lookbacks'].append({
                        'lookback': lookback,
                        'origins': origins,
                        'count': len(origins)
                    })
        
        # Process Experiment 3A: Representation Comparison (L=336, H=192)
        if 'exp3a_representation' in experiments_config:
            exp3a_cfg = experiments_config['exp3a_representation']
            lookback = exp3a_cfg.get('lookback', 336)
            horizon = exp3a_cfg.get('horizon', 192)
            representations = exp3a_cfg.get('representations', ['original', 'differenced'])
            
            manifest['exp3a_representation'] = {}
            
            for split_name, split_df in splits_dict.items():
                origins = self.generate_origins(split_name, split_df, lookback, horizon)
                manifest['exp3a_representation'][split_name] = {
                    'origins': origins,
                    'count': len(origins),
                    'lookback': lookback,
                    'horizon': horizon,
                    'representations': representations
                }
        
        # Process Experiment 3B: Temporal Shift Analysis (uses same origins as 3A)
        if 'exp3b_temporal_shift' in experiments_config:
            exp3b_cfg = experiments_config['exp3b_temporal_shift']
            lookback = exp3b_cfg.get('lookback', 336)
            horizon = exp3b_cfg.get('horizon', 192)
            quarters = exp3b_cfg.get('quarters', ['q1', 'q2', 'q3', 'q4'])
            
            manifest['exp3b_temporal_shift'] = {}
            
            for split_name, split_df in splits_dict.items():
                origins = self.generate_origins(split_name, split_df, lookback, horizon)
                manifest['exp3b_temporal_shift'][split_name] = {
                    'origins': origins,
                    'count': len(origins),
                    'lookback': lookback,
                    'horizon': horizon,
                    'quarters': quarters
                }
        
        return manifest
    
    def save_manifests(
        self,
        output_path: str,
        manifest: Dict[str, Any]
    ) -> None:
        """
        Save origin manifests to JSON file.
        
        Args:
            output_path: Path where manifests should be saved (e.g., 'outputs/artifacts/origin_manifests.json')
            manifest: Manifest dictionary from get_origin_manifest()
            
        Raises:
            IOError: If file cannot be written
            TypeError: If manifest contains non-serializable objects
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        except (IOError, TypeError) as e:
            raise IOError(f"Failed to save manifests to {output_path}: {str(e)}")
    
    @staticmethod
    def verify_origin_consistency(
        manifest1: Dict[str, Any],
        manifest2: Dict[str, Any],
        experiment_key: str = 'exp1_horizon'
    ) -> bool:
        """
        Verify that two manifests use identical origins for a given experiment.
        
        Args:
            manifest1: First origin manifest
            manifest2: Second origin manifest
            experiment_key: Experiment to compare (default: 'exp1_horizon')
            
        Returns:
            bool: True if all splits and configurations have identical origins, False otherwise
            
        Raises:
            KeyError: If experiment_key not found in one or both manifests
        """
        if experiment_key not in manifest1:
            raise KeyError(f"Experiment '{experiment_key}' not found in manifest1")
        
        if experiment_key not in manifest2:
            raise KeyError(f"Experiment '{experiment_key}' not found in manifest2")
        
        exp1 = manifest1[experiment_key]
        exp2 = manifest2[experiment_key]
        
        # Check all split names match
        if set(exp1.keys()) != set(exp2.keys()):
            return False
        
        # For exp1_horizon: compare origin sets across horizons
        if experiment_key == 'exp1_horizon':
            for split_name in exp1.keys():
                h1_horizons = exp1[split_name].get('horizons', [])
                h2_horizons = exp2[split_name].get('horizons', [])
                
                if len(h1_horizons) != len(h2_horizons):
                    return False
                
                for h1_entry, h2_entry in zip(h1_horizons, h2_horizons):
                    if h1_entry['horizon'] != h2_entry['horizon']:
                        return False
                    if h1_entry['origins'] != h2_entry['origins']:
                        return False
        
        # For exp2_lookback: compare origin sets across lookbacks
        elif experiment_key == 'exp2_lookback':
            for split_name in exp1.keys():
                l1_lookbacks = exp1[split_name].get('lookbacks', [])
                l2_lookbacks = exp2[split_name].get('lookbacks', [])
                
                if len(l1_lookbacks) != len(l2_lookbacks):
                    return False
                
                for l1_entry, l2_entry in zip(l1_lookbacks, l2_lookbacks):
                    if l1_entry['lookback'] != l2_entry['lookback']:
                        return False
                    if l1_entry['origins'] != l2_entry['origins']:
                        return False
        
        # For exp3a and exp3b: compare single origin sets per split
        else:
            for split_name in exp1.keys():
                if exp1[split_name].get('origins') != exp2[split_name].get('origins'):
                    return False
        
        return True
