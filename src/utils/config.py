"""Configuration loading, validation, and management."""

import yaml
from typing import Any, Dict, Optional
from pathlib import Path


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML file; raise ValueError if not found or invalid syntax.
    
    Args:
        path: Path to YAML file.
    
    Returns:
        Dictionary containing parsed YAML content.
    
    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If YAML syntax is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate essential keys; return True if valid, raise ValueError otherwise.
    
    Args:
        config: Configuration dictionary to validate.
    
    Returns:
        True if valid.
    
    Raises:
        ValueError: If required keys are missing.
    """
    # Check dataset keys
    if 'dataset' not in config or 'path' not in config['dataset']:
        raise ValueError("Missing dataset.path in config")
    if 'splits' not in config:
        raise ValueError("Missing splits in config")
    if 'training' not in config:
        raise ValueError("Missing training params in config")
    return True


def merge_configs(study_config: Dict, models_config: Dict, experiments_config: Dict) -> Dict:
    """Merge three config dicts into unified config namespace.
    
    Args:
        study_config: Study configuration (dataset, splits, training params).
        models_config: Model architectures.
        experiments_config: Experiment definitions.
    
    Returns:
        Merged configuration dictionary.
    """
    return {
        'dataset': study_config.get('dataset'),
        'splits': study_config.get('splits'),
        'training': study_config.get('training'),
        'lookbacks': study_config.get('lookbacks'),
        'horizons': study_config.get('horizons'),
        'seeds': study_config.get('seeds'),
        'models': models_config,
        'experiments': experiments_config.get('experiments'),
        'device': study_config.get('device', 'auto'),
        'output_dir': study_config.get('output_dir', 'outputs'),
        'precision': study_config.get('precision', 'float32'),
    }


def get_config(config_dir: str | Path = 'configs') -> Dict[str, Any]:
    """Load and merge all three config files; return unified namespace.
    
    Args:
        config_dir: Directory containing study.yaml, models.yaml, experiments.yaml.
    
    Returns:
        Merged configuration dictionary.
    
    Raises:
        FileNotFoundError: If any config file is missing.
        ValueError: If validation fails.
    """
    config_dir = Path(config_dir)
    study = load_yaml(config_dir / 'study.yaml')
    models = load_yaml(config_dir / 'models.yaml')
    experiments = load_yaml(config_dir / 'experiments.yaml')
    
    validate_config(study)
    
    return merge_configs(study, models, experiments)
