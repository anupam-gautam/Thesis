"""Device detection and seed fixing for reproducibility."""

import os
import sys
import random
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def detect_device() -> str:
    """Detect available device: 'cuda' if GPU available, else 'cpu'.
    
    Returns:
        Device string: 'cuda' or 'cpu'.
    """
    if HAS_TORCH and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.
    
    Sets seeds for:
    - Python random module
    - NumPy random number generator
    - PyTorch (if available)
    - PYTHONHASHSEED environment variable
    
    Args:
        seed: Integer seed value.
    """
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # OS-level
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device_string(device_config: str) -> str:
    """Get device string: 'auto' → detect, 'cuda'/'cpu' → use as-is.
    
    Args:
        device_config: Device configuration string.
    
    Returns:
        Device string: 'cuda' or 'cpu'.
    
    Raises:
        ValueError: If device_config is invalid.
    """
    if device_config == 'auto':
        return detect_device()
    elif device_config in ['cuda', 'cpu']:
        return device_config
    else:
        raise ValueError(f"Invalid device config: {device_config}")


def get_device():
    """Return torch.device object if torch available, else None.
    
    Returns:
        torch.device object or None if PyTorch not available.
    """
    if HAS_TORCH:
        device_str = detect_device()
        return torch.device(device_str)
    return None
