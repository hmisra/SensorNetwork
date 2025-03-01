"""
Integration tests for the example workflows.
This ensures the example code provided in the examples/ directory
continues to work as expected.
"""
import pytest
import torch
import sys
import os
from pathlib import Path
import importlib.util
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt  # Add this line to explicitly import pyplot

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor_actuator_network import SensorAugmentor, set_seed


def import_module_from_path(module_name, file_path):
    """Dynamically import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestExampleWorkflows:
    """Integration tests for example workflows."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        set_seed(42)  # For reproducibility
        
        # Path to examples
        examples_dir = Path(__file__).parent.parent.parent / 'examples'
        
        # Create a temporary directory for output files
        temp_dir = tempfile.mkdtemp()
        
        return {
            "examples_dir": examples_dir,
            "temp_dir": temp_dir
        }
    
    def teardown_method(self, method):
        """Clean up temporary files after each test."""
        # If there's a temp_dir in the test's namespace, clean it up
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
    
    def test_custom_dataset_example(self, setup, monkeypatch):
        """Test the custom_dataset_example.py workflow."""
        examples_dir = setup["examples_dir"]
        temp_dir = setup["temp_dir"]
        
        # Path to the example script
        example_path = examples_dir / 'custom_dataset_example.py'
        
        # Mock figure saving to avoid GUI
        def mock_savefig(path, **kwargs):
            pass
        
        monkeypatch.setattr(plt.Figure, 'savefig', mock_savefig)  # Use plt instead of matplotlib.pyplot
        
        # Modify sys.argv to pass arguments to the script
        orig_argv = sys.argv
        sys.argv = [
            str(example_path),
            '--epochs', '2',  # Reduce epochs for faster testing
            '--batch_size', '16',
            '--no_plot'  # Skip plotting to avoid GUI issues in testing
        ]
        
        try:
            # Import and run the example module
            example_module = import_module_from_path(
                'custom_dataset_example',
                str(example_path)
            )
            
            # Check if a main function exists and call it
            if hasattr(example_module, 'main'):
                # Create a temporary output path
                output_path = os.path.join(temp_dir, 'model.pt')
                
                # Run main with test parameters
                result = example_module.main(
                    num_samples=50,
                    epochs=2,
                    batch_size=16,
                    model_output_path=output_path,
                    no_plot=True
                )
                
                # Check that main executed without errors
                assert result is not None
                
                # Check that the model was saved
                assert os.path.exists(output_path)
                
                # Try loading the model to confirm it's valid
                loaded_data = torch.load(output_path)
                assert "model_state" in loaded_data
                assert "metadata" in loaded_data
        finally:
            # Restore original sys.argv
            sys.argv = orig_argv
    
    def test_time_series_example(self, setup, monkeypatch):
        """Test the time_series_example.py workflow."""
        examples_dir = setup["examples_dir"]
        temp_dir = setup["temp_dir"]
        
        # Path to the example script
        example_path = examples_dir / 'time_series_example.py'
        
        # Mock figure saving to avoid GUI
        def mock_savefig(path, **kwargs):
            pass
        
        monkeypatch.setattr(plt.Figure, 'savefig', mock_savefig)  # Use plt instead of matplotlib.pyplot
        
        # Modify sys.argv to pass arguments to the script
        orig_argv = sys.argv
        sys.argv = [
            str(example_path),
            '--epochs', '2',  # Reduce epochs for faster testing
            '--batch_size', '16',
            '--no_plot'  # Skip plotting to avoid GUI issues in testing
        ]
        
        try:
            # Import and run the example module
            example_module = import_module_from_path(
                'time_series_example',
                str(example_path)
            )
            
            # Check if a main function exists and call it
            if hasattr(example_module, 'main'):
                # Create a temporary output path
                output_path = os.path.join(temp_dir, 'vibration_model.pt')
                
                # Run main with test parameters
                result = example_module.main(
                    num_samples=50,
                    seq_length=64,  # Shorter sequence for faster testing
                    epochs=2,
                    batch_size=16,
                    model_output_path=output_path,
                    no_plot=True
                )
                
                # Check that main executed without errors
                assert result is not None
                
                # Check that the model was saved
                assert os.path.exists(output_path)
                
                # Try loading the model to confirm it's valid
                loaded_data = torch.load(output_path)
                assert "model_state" in loaded_data
                assert "metadata" in loaded_data
        finally:
            # Restore original sys.argv
            sys.argv = orig_argv
    
    def test_custom_dataset_creation(self, setup):
        """Test creation and functionality of the custom datasets in examples."""
        examples_dir = setup["examples_dir"]
        
        # Import the custom dataset classes from examples
        env_dataset_module = import_module_from_path(
            'env_dataset',
            str(examples_dir / 'custom_dataset_example.py')
        )
        
        vib_dataset_module = import_module_from_path(
            'vib_dataset',
            str(examples_dir / 'time_series_example.py')
        )
        
        # Create instances of the custom datasets
        env_dataset = env_dataset_module.EnvironmentalSensorDataset(
            num_samples=20,
            time_steps=12
        )
        
        vib_dataset = vib_dataset_module.VibrationSensorDataset(
            num_samples=20,
            seq_length=64,
            num_sensors=3
        )
        
        # Test basic dataset functionality
        assert len(env_dataset) == 20
        assert len(vib_dataset) == 20
        
        # Test getitem
        env_sample = env_dataset[0]
        vib_sample = vib_dataset[0]
        
        # Check that they return the expected tuple structure (x_lq, x_hq, y_cmd)
        assert len(env_sample) == 3
        assert len(vib_sample) == 3
        
        # Check that all tensors have the expected shape
        assert env_sample[0].ndim == 1  # x_lq
        assert env_sample[1].ndim == 1  # x_hq
        assert env_sample[2].ndim == 1  # y_cmd
        
        # Check normalization attributes
        assert hasattr(env_dataset, 'mean_lq')
        assert hasattr(env_dataset, 'std_lq')
        assert hasattr(vib_dataset, 'mean_lq')
        assert hasattr(vib_dataset, 'std_lq') 