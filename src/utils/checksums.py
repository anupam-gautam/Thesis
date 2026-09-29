"""Compute checksums for data governance and reproducibility."""

import hashlib
from pathlib import Path
from typing import Dict, Any
import json


def md5_file(path: str | Path, chunk_size: int = 8192) -> str:
    """Compute MD5 checksum of file.
    
    Args:
        path: Path to file.
        chunk_size: Size of chunks to read (default 8192 bytes).
    
    Returns:
        MD5 checksum as hex string (32 characters).
    
    Raises:
        FileNotFoundError: If file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def md5_dict(d: Dict[str, Any]) -> str:
    """Compute MD5 checksum of dict (JSON-serialized, sorted keys).
    
    Args:
        d: Dictionary to hash.
    
    Returns:
        MD5 checksum as hex string (32 characters).
    """
    json_str = json.dumps(d, sort_keys=True, default=str)
    return hashlib.md5(json_str.encode()).hexdigest()


def compute_run_id(config: Dict, dataset_checksum: str, preprocessing_checksum: str,
                   origins_checksum: str, seed: int) -> str:
    """Compute unique run ID: SHA-256 hash of combined checksums.
    
    Produces deterministic 16-character hex identifier for run reproducibility.
    
    Args:
        config: Configuration dictionary.
        dataset_checksum: MD5 checksum of dataset file.
        preprocessing_checksum: MD5 checksum of preprocessing settings.
        origins_checksum: MD5 checksum of origin manifest.
        seed: Random seed value.
    
    Returns:
        16-character SHA-256 hash hex string.
    """
    config_checksum = md5_dict(config)
    combined = f"{config_checksum}_{dataset_checksum}_{preprocessing_checksum}_{origins_checksum}_{seed}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
