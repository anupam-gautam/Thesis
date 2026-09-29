"""
Generate preprocessing artifacts for Task 2.5.

This script:
1. Loads data using DataLoader
2. Splits data using DataSplitter
3. Fits preprocessors on training data
4. Applies preprocessing to all splits (Branch O and D)
5. Saves preprocessed DataFrames
6. Generates preprocessing config and checksums
"""

import os
import sys
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.loader import DataLoader
from data.splits import DataSplitter
from data.preprocessor import Preprocessor
from data.windows import WindowGenerator
from utils.checksums import md5_file, md5_dict





def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def main():
    """Main execution function."""
    logger = setup_logging()
    
    # Load configuration
    config_path = project_root / "configs" / "study.yaml"
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Dataset path: {config['dataset']['path']}")
    
    # Setup paths
    dataset_path = project_root / config['dataset']['path']
    output_dir = project_root / config['output_dir']
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    logger.info("=" * 80)
    logger.info("STEP 1: Loading data")
    logger.info("=" * 80)
    
    loader = DataLoader(str(dataset_path))
    df = loader.load()
    logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Date range: {df[config['dataset']['timestamp_col']].min()} to {df[config['dataset']['timestamp_col']].max()}")
    
    # Compute dataset checksum
    dataset_md5 = md5_file(str(dataset_path))
    logger.info(f"Dataset MD5: {dataset_md5}")
    
    # Step 2: Split data
    logger.info("=" * 80)
    logger.info("STEP 2: Splitting data")
    logger.info("=" * 80)
    
    timestamp_col = config['dataset']['timestamp_col']
    target_col = config['dataset']['target_col']
    channel_cols = config['dataset']['channels']
    
    # Numeric columns: target + channels
    numeric_cols = [target_col] + channel_cols
    
    splitter = DataSplitter(
        df,
        timestamp_col=timestamp_col,
        target_col=target_col,
        train_end=config['splits']['train_end'],
        val_end=config['splits']['val_end']
    )
    
    df_train, df_val, df_test = splitter.split()
    split_info = splitter.get_split_info()
    
    logger.info(f"Train: {len(df_train)} rows ({split_info['train']['start_date']} to {split_info['train']['end_date']})")
    logger.info(f"Val:   {len(df_val)} rows ({split_info['validation']['start_date']} to {split_info['validation']['end_date']})")
    logger.info(f"Test:  {len(df_test)} rows ({split_info['test']['start_date']} to {split_info['test']['end_date']})")
    
    # Step 3: Fit preprocessors
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Fitting preprocessors")
    logger.info("=" * 80)
    
    logger.info(f"Target column: {target_col}")
    logger.info(f"Channel columns: {channel_cols}")
    
    preprocessor = Preprocessor(df, timestamp_col, numeric_cols)
    scalers = preprocessor.fit_scalers(df_train)
    logger.info(f"Scalers fitted for branches: {list(scalers.keys())}")
    
    # Step 4: Apply preprocessing
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Applying preprocessing")
    logger.info("=" * 80)
    
    # Branch O preprocessing (original representation)
    logger.info("Branch O (Original) preprocessing...")
    df_train_o = preprocessor.forward_branch_o(df_train)
    df_val_o = preprocessor.forward_branch_o(df_val)
    df_test_o = preprocessor.forward_branch_o(df_test)
    
    # Add timestamp column to preprocessed data
    df_train_o_full = pd.concat([df_train[[timestamp_col]], df_train_o], axis=1)
    df_val_o_full = pd.concat([df_val[[timestamp_col]], df_val_o], axis=1)
    df_test_o_full = pd.concat([df_test[[timestamp_col]], df_test_o], axis=1)
    
    logger.info(f"  Train O: {len(df_train_o_full)} rows")
    logger.info(f"  Val O:   {len(df_val_o_full)} rows")
    logger.info(f"  Test O:  {len(df_test_o_full)} rows")
    
    # Branch D preprocessing (differenced representation)
    logger.info("Branch D (Differenced) preprocessing...")
    df_train_d, first_level_train = preprocessor.forward_branch_d(df_train)
    df_val_d, first_level_val = preprocessor.forward_branch_d(df_val)
    df_test_d, first_level_test = preprocessor.forward_branch_d(df_test)
    
    # Add timestamp column to preprocessed data (skip first row due to differencing)
    df_train_d_full = pd.concat([df_train[[timestamp_col]].iloc[1:].reset_index(drop=True), 
                                  df_train_d.reset_index(drop=True)], axis=1)
    df_val_d_full = pd.concat([df_val[[timestamp_col]].iloc[1:].reset_index(drop=True), 
                                df_val_d.reset_index(drop=True)], axis=1)
    df_test_d_full = pd.concat([df_test[[timestamp_col]].iloc[1:].reset_index(drop=True), 
                                 df_test_d.reset_index(drop=True)], axis=1)
    
    logger.info(f"  Train D: {len(df_train_d_full)} rows")
    logger.info(f"  Val D:   {len(df_val_d_full)} rows")
    logger.info(f"  Test D:  {len(df_test_d_full)} rows")
    
    # Step 5: Save preprocessed data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Saving preprocessed data")
    logger.info("=" * 80)
    
    # Save Branch O
    branch_o_path = artifacts_dir / "preprocessed_branch_o.csv"
    df_combined_o = pd.concat([df_train_o_full, df_val_o_full, df_test_o_full], ignore_index=False)
    df_combined_o.to_csv(branch_o_path, index=False)
    logger.info(f"Saved Branch O: {branch_o_path}")
    logger.info(f"  Rows: {len(df_combined_o)}, Size: {branch_o_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save Branch D
    branch_d_path = artifacts_dir / "preprocessed_branch_d.csv"
    df_combined_d = pd.concat([df_train_d_full, df_val_d_full, df_test_d_full], ignore_index=False)
    df_combined_d.to_csv(branch_d_path, index=False)
    logger.info(f"Saved Branch D: {branch_d_path}")
    logger.info(f"  Rows: {len(df_combined_d)}, Size: {branch_d_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Step 6: Generate origin manifests
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Generating origin manifests")
    logger.info("=" * 80)
    
    # Step 6: Generate origin manifests
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Generating origin manifests")
    logger.info("=" * 80)
    
    # Get max lookback and horizon
    lookback_max = max(config['lookbacks'])
    horizon_max = max(config['horizons'])
    
    # Create WindowGenerator with combined dataframe
    df_combined = pd.concat([df_train, df_val, df_test], ignore_index=False)
    window_gen = WindowGenerator(
        df_combined,
        timestamp_col=timestamp_col,
        lookback_max=lookback_max,
        horizon_max=horizon_max
    )
    
    # Generate manifests for key experiments
    manifests = {}
    for lookback in config['lookbacks']:
        for horizon in config['horizons']:
            exp_key = f"L{lookback}_H{horizon}"
            
            origins_train = window_gen.generate_origins('train', df_train, lookback, horizon)
            origins_val = window_gen.generate_origins('val', df_val, lookback, horizon)
            origins_test = window_gen.generate_origins('test', df_test, lookback, horizon)
            
            manifests[exp_key] = {
                'lookback': lookback,
                'horizon': horizon,
                'train_origins_count': len(origins_train),
                'val_origins_count': len(origins_val),
                'test_origins_count': len(origins_test),
                'total_origins': len(origins_train) + len(origins_val) + len(origins_test)
            }
            
            logger.info(f"  {exp_key}: Train={len(origins_train)}, Val={len(origins_val)}, Test={len(origins_test)}")
    
    # Step 7: Create preprocessing config
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Creating preprocessing configuration")
    logger.info("=" * 80)
    
    preprocessing_config = {
        'preprocessing_method': 'standard_scaling',
        'scaler_type': 'StandardScaler',
        'scaler_params': {
            'branch_o': {
                'mean': preprocessor.scaler_params['branch_o']['mean'].tolist(),
                'scale': preprocessor.scaler_params['branch_o']['scale'].tolist()
            },
            'branch_d': {
                'mean': preprocessor.scaler_params['branch_d']['mean'].tolist(),
                'scale': preprocessor.scaler_params['branch_d']['scale'].tolist()
            }
        },
        'split_dates': {
            'train_start': config['splits']['train_start'],
            'train_end': config['splits']['train_end'],
            'val_end': config['splits']['val_end']
        },
        'representation_branches': ['branch_o', 'branch_d'],
        'timestamp_col': timestamp_col,
        'target_col': target_col,
        'numeric_cols': numeric_cols,
        'train_rows': len(df_train),
        'val_rows': len(df_val),
        'test_rows': len(df_test)
    }
    
    preprocessing_config_path = artifacts_dir / "preprocessing_config.json"
    with open(preprocessing_config_path, 'w') as f:
        json.dump(preprocessing_config, f, indent=2)
    logger.info(f"Saved preprocessing config: {preprocessing_config_path}")
    
    # Step 8: Compute checksums
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Computing checksums")
    logger.info("=" * 80)
    
    # Dataset MD5
    dataset_md5 = md5_file(str(dataset_path))
    
    # Preprocessing config MD5
    preprocessing_md5 = md5_dict(preprocessing_config)
    
    logger.info(f"Dataset MD5:       {dataset_md5}")
    logger.info(f"Preprocessing MD5: {preprocessing_md5}")
    
    # Step 9: Save data checksums
    logger.info("\n" + "=" * 80)
    logger.info("STEP 9: Saving data checksums")
    logger.info("=" * 80)
    
    data_checksums = {
        'dataset_md5': dataset_md5,
        'preprocessing_md5': preprocessing_md5,
        'dataset_path': str(dataset_path),
        'creation_timestamp': datetime.now().isoformat(),
        'artifacts': {
            'preprocessed_branch_o': {
                'path': str(branch_o_path),
                'rows': len(df_combined_o),
                'size_mb': branch_o_path.stat().st_size / 1024 / 1024
            },
            'preprocessed_branch_d': {
                'path': str(branch_d_path),
                'rows': len(df_combined_d),
                'size_mb': branch_d_path.stat().st_size / 1024 / 1024
            },
            'scalers_branch_o': {
                'path': str(artifacts_dir / "scalers_branch_o.pkl"),
                'exists': (artifacts_dir / "scalers_branch_o.pkl").exists()
            },
            'scalers_branch_d': {
                'path': str(artifacts_dir / "scalers_branch_d.pkl"),
                'exists': (artifacts_dir / "scalers_branch_d.pkl").exists()
            },
            'origin_manifests': {
                'path': str(artifacts_dir / "origin_manifests.json"),
                'exists': (artifacts_dir / "origin_manifests.json").exists()
            }
        },
        'origin_summary': manifests
    }
    
    checksums_path = artifacts_dir / "data_checksums.json"
    with open(checksums_path, 'w') as f:
        json.dump(data_checksums, f, indent=2)
    logger.info(f"Saved data checksums: {checksums_path}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING ARTIFACTS GENERATION COMPLETE")
    logger.info("=" * 80)
    
    logger.info("\nGenerated Artifacts:")
    logger.info(f"  ✓ preprocessed_branch_o.csv ({len(df_combined_o)} rows, {branch_o_path.stat().st_size / 1024 / 1024:.2f} MB)")
    logger.info(f"  ✓ preprocessed_branch_d.csv ({len(df_combined_d)} rows, {branch_d_path.stat().st_size / 1024 / 1024:.2f} MB)")
    logger.info(f"  ✓ scalers_branch_o.pkl (exists: {(artifacts_dir / 'scalers_branch_o.pkl').exists()})")
    logger.info(f"  ✓ scalers_branch_d.pkl (exists: {(artifacts_dir / 'scalers_branch_d.pkl').exists()})")
    logger.info(f"  ✓ origin_manifests.json (exists: {(artifacts_dir / 'origin_manifests.json').exists()})")
    logger.info(f"  ✓ preprocessing_config.json")
    logger.info(f"  ✓ data_checksums.json")
    
    logger.info(f"\nOrigin Manifest Summary:")
    for exp_key, summary in manifests.items():
        logger.info(f"  {exp_key}: Train={summary['train_origins_count']}, Val={summary['val_origins_count']}, Test={summary['test_origins_count']} (Total: {summary['total_origins']})")
    
    logger.info(f"\nAll artifacts saved to: {artifacts_dir}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
