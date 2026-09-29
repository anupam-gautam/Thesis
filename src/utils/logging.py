"""Experiment logging and run status tracking."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List


class RunLogger:
    """Log experiment runs and maintain run manifest.
    
    Manages persistent tracking of experiment runs with status lifecycle:
    pending → running → completed/failed
    """
    
    def __init__(self, manifest_path: str | Path = 'outputs/runs/run_manifest.json'):
        """Initialize RunLogger.
        
        Args:
            manifest_path: Path to run manifest JSON file.
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load run manifest if exists, else return empty dict.
        
        Returns:
            Dictionary with 'runs' key containing list of run entries.
        """
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {'runs': []}
    
    def add_run(self, run_id: str, config: Dict, status: str = 'pending') -> None:
        """Add or update run in manifest.
        
        Args:
            run_id: Unique run identifier.
            config: Configuration dictionary for this run.
            status: Initial status (default: 'pending').
        """
        run_entry = {
            'run_id': run_id,
            'config': config,
            'status': status,
            'timestamp_created': datetime.utcnow().isoformat(),
            'timestamp_started': None,
            'timestamp_completed': None,
            'error': None,
        }
        
        # Check if run_id already exists
        existing = next((r for r in self.manifest['runs'] if r['run_id'] == run_id), None)
        if existing:
            existing.update(run_entry)
        else:
            self.manifest['runs'].append(run_entry)
        
        self._save_manifest()
    
    def update_run(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        """Update run status (pending → running → completed/failed).
        
        Args:
            run_id: Unique run identifier.
            status: New status value.
            error: Error message if status is 'failed' (optional).
        
        Raises:
            ValueError: If run_id not found.
        """
        run = next((r for r in self.manifest['runs'] if r['run_id'] == run_id), None)
        if not run:
            raise ValueError(f"Run ID not found: {run_id}")
        
        run['status'] = status
        if status == 'running':
            run['timestamp_started'] = datetime.utcnow().isoformat()
        elif status in ['completed', 'failed']:
            run['timestamp_completed'] = datetime.utcnow().isoformat()
        
        if error:
            run['error'] = error
        
        self._save_manifest()
    
    def _save_manifest(self) -> None:
        """Save manifest to JSON file."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def get_completed_runs(self) -> List[str]:
        """Return list of completed run IDs.
        
        Returns:
            List of run IDs with status 'completed'.
        """
        return [r['run_id'] for r in self.manifest['runs'] if r['status'] == 'completed']
    
    def get_failed_runs(self) -> List[Dict]:
        """Return list of failed runs with error messages.
        
        Returns:
            List of run entries with status 'failed'.
        """
        return [r for r in self.manifest['runs'] if r['status'] == 'failed']
