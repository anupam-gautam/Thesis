"""Unit tests for utility modules."""

import tempfile
import yaml
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import config, device, checksums, logging as exp_logging


class TestConfig:
    """Tests for config.py module."""
    
    def test_load_yaml_valid(self):
        """Test YAML loading with valid file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'key': 'value', 'number': 42, 'nested': {'inner': 'data'}}, f)
            temp_path = f.name
        
        try:
            data = config.load_yaml(temp_path)
            assert data['key'] == 'value'
            assert data['number'] == 42
            assert data['nested']['inner'] == 'data'
        finally:
            Path(temp_path).unlink()
    
    def test_load_yaml_missing_file(self):
        """Test YAML loading with missing file raises FileNotFoundError."""
        try:
            config.load_yaml('/nonexistent/path/file.yaml')
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "not found" in str(e).lower()
    
    def test_load_yaml_invalid_syntax(self):
        """Test YAML loading with invalid syntax raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: syntax: [")
            temp_path = f.name
        
        try:
            config.load_yaml(temp_path)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid yaml" in str(e).lower()
        finally:
            Path(temp_path).unlink()
    
    def test_validate_config_valid(self):
        """Test config validation with valid config."""
        valid_config = {
            'dataset': {'path': '/path/to/data.csv'},
            'splits': {'train_end': '2023-01-01'},
            'training': {'batch_size': 32}
        }
        assert config.validate_config(valid_config) == True
    
    def test_validate_config_missing_dataset_path(self):
        """Test config validation raises ValueError for missing dataset.path."""
        invalid_config = {
            'dataset': {},
            'splits': {'train_end': '2023-01-01'},
            'training': {'batch_size': 32}
        }
        try:
            config.validate_config(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "dataset.path" in str(e)
    
    def test_validate_config_missing_splits(self):
        """Test config validation raises ValueError for missing splits."""
        invalid_config = {
            'dataset': {'path': '/path/to/data.csv'},
            'training': {'batch_size': 32}
        }
        try:
            config.validate_config(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "splits" in str(e)
    
    def test_validate_config_missing_training(self):
        """Test config validation raises ValueError for missing training."""
        invalid_config = {
            'dataset': {'path': '/path/to/data.csv'},
            'splits': {'train_end': '2023-01-01'},
        }
        try:
            config.validate_config(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "training" in str(e)
    
    def test_merge_configs(self):
        """Test config merging."""
        study = {
            'dataset': {'path': 'data.csv'},
            'splits': {'train_end': '2023-01-01'},
            'training': {'batch_size': 32},
            'lookbacks': [96, 336],
            'horizons': [96, 192],
            'seeds': [42, 123],
            'device': 'cuda',
            'output_dir': 'results'
        }
        models = {
            'dlinear': {'d_model': 128},
            'nlinear': {'d_model': 128}
        }
        experiments = {
            'experiments': {
                'exp1': {'name': 'Experiment 1'}
            }
        }
        
        merged = config.merge_configs(study, models, experiments)
        
        assert merged['dataset'] == study['dataset']
        assert merged['models'] == models
        assert merged['experiments'] == experiments['experiments']
        assert merged['device'] == 'cuda'
        assert merged['output_dir'] == 'results'
        assert merged['lookbacks'] == [96, 336]
        assert merged['seeds'] == [42, 123]


class TestDevice:
    """Tests for device.py module."""
    
    def test_detect_device_returns_string(self):
        """Test detect_device returns a valid string."""
        dev = device.detect_device()
        assert isinstance(dev, str)
        assert dev in ['cuda', 'cpu']
    
    def test_set_seed_reproducibility(self):
        """Test seed fixing produces deterministic random values."""
        import random
        import numpy as np
        
        # First run with seed 42
        device.set_seed(42)
        val1_py = random.random()
        val1_np = np.random.rand()
        
        # Second run with seed 42
        device.set_seed(42)
        val2_py = random.random()
        val2_np = np.random.rand()
        
        assert val1_py == val2_py, "Python random not reproducible"
        assert val1_np == val2_np, "NumPy random not reproducible"
    
    def test_set_seed_different_seeds(self):
        """Test different seeds produce different values."""
        import random
        
        device.set_seed(42)
        val1 = random.random()
        
        device.set_seed(123)
        val2 = random.random()
        
        assert val1 != val2, "Different seeds should produce different values"
    
    def test_get_device_string_auto(self):
        """Test get_device_string with 'auto' returns valid device."""
        dev_str = device.get_device_string('auto')
        assert dev_str in ['cuda', 'cpu']
    
    def test_get_device_string_explicit(self):
        """Test get_device_string with explicit device."""
        assert device.get_device_string('cpu') == 'cpu'
        assert device.get_device_string('cuda') == 'cuda'
    
    def test_get_device_string_invalid(self):
        """Test get_device_string raises ValueError for invalid device."""
        try:
            device.get_device_string('tpu')
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid" in str(e).lower()
    
    def test_get_device(self):
        """Test get_device returns torch.device or None."""
        dev = device.get_device()
        # Will be torch.device or None depending on PyTorch availability
        if dev is not None:
            assert str(dev) in ['cpu', 'cuda:0', 'cuda']


class TestChecksums:
    """Tests for checksums.py module."""
    
    def test_md5_file(self):
        """Test file MD5 computation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            checksum = checksums.md5_file(temp_path)
            assert isinstance(checksum, str)
            assert len(checksum) == 32  # MD5 hex is 32 chars
        finally:
            Path(temp_path).unlink()
    
    def test_md5_file_missing(self):
        """Test MD5 file computation with missing file raises FileNotFoundError."""
        try:
            checksums.md5_file('/nonexistent/file.txt')
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "not found" in str(e).lower()
    
    def test_md5_file_reproducible(self):
        """Test MD5 computation is reproducible."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            checksum1 = checksums.md5_file(temp_path)
            checksum2 = checksums.md5_file(temp_path)
            assert checksum1 == checksum2
        finally:
            Path(temp_path).unlink()
    
    def test_md5_dict(self):
        """Test dict MD5 computation."""
        d = {'key': 'value', 'number': 42}
        checksum = checksums.md5_dict(d)
        assert isinstance(checksum, str)
        assert len(checksum) == 32
    
    def test_md5_dict_order_invariant(self):
        """Test MD5 dict computation is order-invariant (sorted keys)."""
        d1 = {'a': 1, 'b': 2, 'c': 3}
        d2 = {'c': 3, 'a': 1, 'b': 2}
        
        checksum1 = checksums.md5_dict(d1)
        checksum2 = checksums.md5_dict(d2)
        assert checksum1 == checksum2
    
    def test_compute_run_id(self):
        """Test run ID computation."""
        cfg = {'model': 'dlinear', 'horizon': 192}
        run_id = checksums.compute_run_id(cfg, 'abc123', 'def456', 'ghi789', 42)
        
        assert isinstance(run_id, str)
        assert len(run_id) == 16
        assert all(c in '0123456789abcdef' for c in run_id)
    
    def test_compute_run_id_deterministic(self):
        """Test run ID is deterministic."""
        cfg = {'model': 'dlinear', 'horizon': 192}
        run_id1 = checksums.compute_run_id(cfg, 'abc123', 'def456', 'ghi789', 42)
        run_id2 = checksums.compute_run_id(cfg, 'abc123', 'def456', 'ghi789', 42)
        assert run_id1 == run_id2
    
    def test_compute_run_id_different_params(self):
        """Test different params produce different run IDs."""
        cfg = {'model': 'dlinear', 'horizon': 192}
        run_id1 = checksums.compute_run_id(cfg, 'abc123', 'def456', 'ghi789', 42)
        run_id2 = checksums.compute_run_id(cfg, 'abc123', 'def456', 'ghi789', 123)
        assert run_id1 != run_id2


class TestRunLogger:
    """Tests for logging.py module."""
    
    def test_run_logger_creation(self):
        """Test RunLogger initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = exp_logging.RunLogger(Path(tmpdir) / 'manifest.json')
            assert logger.manifest_path.exists()
            assert 'runs' in logger.manifest
    
    def test_add_run(self):
        """Test adding a run to manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = exp_logging.RunLogger(Path(tmpdir) / 'manifest.json')
            
            cfg = {'model': 'dlinear', 'horizon': 192}
            logger.add_run('run_123', cfg, 'pending')
            
            assert len(logger.manifest['runs']) == 1
            assert logger.manifest['runs'][0]['run_id'] == 'run_123'
            assert logger.manifest['runs'][0]['status'] == 'pending'
    
    def test_update_run_to_running(self):
        """Test updating run status to running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = exp_logging.RunLogger(Path(tmpdir) / 'manifest.json')
            
            cfg = {'model': 'dlinear', 'horizon': 192}
            logger.add_run('run_123', cfg, 'pending')
            logger.update_run('run_123', 'running')
            
            run = logger.manifest['runs'][0]
            assert run['status'] == 'running'
            assert run['timestamp_started'] is not None
    
    def test_update_run_to_completed(self):
        """Test updating run status to completed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = exp_logging.RunLogger(Path(tmpdir) / 'manifest.json')
            
            cfg = {'model': 'dlinear', 'horizon': 192}
            logger.add_run('run_123', cfg, 'pending')
            logger.update_run('run_123', 'completed')
            
            run = logger.manifest['runs'][0]
            assert run['status'] == 'completed'
            assert run['timestamp_completed'] is not None
    
    def test_update_run_with_error(self):
        """Test updating run with error message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = exp_logging.RunLogger(Path(tmpdir) / 'manifest.json')
            
            cfg = {'model': 'dlinear', 'horizon': 192}
            logger.add_run('run_123', cfg, 'pending')
            logger.update_run('run_123', 'failed', 'Out of memory')
            
            run = logger.manifest['runs'][0]
            assert run['status'] == 'failed'
            assert run['error'] == 'Out of memory'
    
    def test_update_run_not_found(self):
        """Test updating non-existent run raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = exp_logging.RunLogger(Path(tmpdir) / 'manifest.json')
            
            try:
                logger.update_run('nonexistent', 'running')
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "not found" in str(e).lower()
    
    def test_get_completed_runs(self):
        """Test retrieving completed run IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = exp_logging.RunLogger(Path(tmpdir) / 'manifest.json')
            
            cfg = {'model': 'dlinear', 'horizon': 192}
            logger.add_run('run_123', cfg, 'pending')
            logger.add_run('run_456', cfg, 'pending')
            
            logger.update_run('run_123', 'completed')
            
            completed = logger.get_completed_runs()
            assert completed == ['run_123']
    
    def test_get_failed_runs(self):
        """Test retrieving failed runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = exp_logging.RunLogger(Path(tmpdir) / 'manifest.json')
            
            cfg = {'model': 'dlinear', 'horizon': 192}
            logger.add_run('run_123', cfg, 'pending')
            logger.add_run('run_456', cfg, 'pending')
            
            logger.update_run('run_123', 'failed', 'Error message')
            
            failed = logger.get_failed_runs()
            assert len(failed) == 1
            assert failed[0]['run_id'] == 'run_123'
            assert failed[0]['error'] == 'Error message'
    
    def test_manifest_persistence(self):
        """Test manifest is saved and reloaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / 'manifest.json'
            
            # Create and add run
            logger1 = exp_logging.RunLogger(manifest_path)
            cfg = {'model': 'dlinear', 'horizon': 192}
            logger1.add_run('run_123', cfg, 'pending')
            logger1.update_run('run_123', 'completed')
            
            # Reload and verify
            logger2 = exp_logging.RunLogger(manifest_path)
            assert len(logger2.manifest['runs']) == 1
            assert logger2.manifest['runs'][0]['run_id'] == 'run_123'
            assert logger2.manifest['runs'][0]['status'] == 'completed'


# Run tests if executed directly
if __name__ == '__main__':
    import sys
    
    # Simple test runner
    test_classes = [TestConfig, TestDevice, TestChecksums, TestRunLogger]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for test_method in test_methods:
            total_tests += 1
            try:
                getattr(test_instance, test_method)()
                print(f"✓ {test_method}")
                passed_tests += 1
            except Exception as e:
                print(f"✗ {test_method}: {e}")
                failed_tests.append((test_class.__name__, test_method, str(e)))
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print('='*60)
    
    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)
