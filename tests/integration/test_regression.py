"""
Regression tests to ensure performance doesn't degrade over time.
These tests compare the current model performance against stored baseline values.
"""
import pytest
import torch
import json
import os
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor_actuator_network import (
    SensorAugmentor, 
    SyntheticSensorDataset, 
    train_model, 
    set_seed
)


class TestRegression:
    """Tests to catch performance regressions."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        set_seed(42)  # For reproducibility
        
        # Create a small model and dataset for quick testing
        model = SensorAugmentor(
            sensor_dim=16,
            hidden_dim=32,
            output_dim=1,
            num_resblocks=1
        )
        
        dataset = SyntheticSensorDataset(
            num_samples=100,
            sensor_dim=16,
            noise_factor=0.3
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Define baseline values (these would normally be stored in a file)
        # For this example, we'll hard-code reasonable values
        baseline = {
            "reconstruction_mse": 0.25,
            "actuator_mse": 0.20,
            "inference_time_ms": 2.0,  # milliseconds per sample
            "num_parameters": 3000
        }
        
        # Create file path for saving/loading baseline values
        baseline_path = Path(__file__).parent / "regression_baseline.json"
        
        return {
            "model": model,
            "dataset": dataset,
            "train_loader": train_loader,
            "baseline": baseline,
            "baseline_path": baseline_path
        }
    
    def _get_or_create_baseline(self, current_metrics, baseline_path):
        """Get existing baseline or create one if it doesn't exist."""
        if os.path.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baseline = json.load(f)
                
            # Ensure all needed keys are in the baseline
            # If not, update with current metrics for those keys
            for key in current_metrics:
                if key not in baseline:
                    baseline[key] = current_metrics[key]
                    # Save the updated baseline
                    with open(baseline_path, "w") as f:
                        json.dump(baseline, f, indent=2)
            
            return baseline
        else:
            # If no baseline exists, create one from current metrics
            with open(baseline_path, "w") as f:
                json.dump(current_metrics, f, indent=2)
            print(f"Created new baseline at {baseline_path}")
            return current_metrics
    
    @pytest.mark.benchmark
    def test_model_size_regression(self, setup):
        """Test that model size doesn't unexpectedly increase."""
        model = setup["model"]
        baseline = setup["baseline"]
        baseline_path = setup["baseline_path"]
        
        # Get current number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        # Get baseline value
        current_metrics = {"num_parameters": num_params}
        baseline = self._get_or_create_baseline(current_metrics, baseline_path)
        
        # Check for regression
        baseline_params = baseline["num_parameters"]
        
        # Log information
        print(f"\nModel Size Regression Test:")
        print(f"Current parameters: {num_params}")
        print(f"Baseline parameters: {baseline_params}")
        
        # Allow for small increases (5% tolerance)
        max_allowed = baseline_params * 1.05
        assert num_params <= max_allowed, (
            f"Model size has increased significantly: {num_params} parameters vs. "
            f"baseline {baseline_params} (max allowed: {max_allowed})"
        )
    
    @pytest.mark.benchmark
    def test_inference_speed_regression(self, setup):
        """Test that inference speed doesn't degrade."""
        model = setup["model"]
        baseline = setup["baseline"]
        baseline_path = setup["baseline_path"]
        
        # Create test batch
        batch_size = 10
        x_test = torch.randn(batch_size, 16)
        
        # Warm up
        for _ in range(5):
            with torch.no_grad():
                model(x_test)
        
        # Measure inference time
        iterations = 50
        import time
        start_time = time.time()
        
        for _ in range(iterations):
            with torch.no_grad():
                model(x_test)
        
        total_time = time.time() - start_time
        avg_time_ms = (total_time / iterations / batch_size) * 1000  # ms per sample
        
        # Get baseline value
        current_metrics = {"inference_time_ms": avg_time_ms}
        baseline = self._get_or_create_baseline(current_metrics, baseline_path)
        
        # Check for regression
        baseline_time = baseline["inference_time_ms"]
        
        # Log information
        print(f"\nInference Speed Regression Test:")
        print(f"Current inference time: {avg_time_ms:.4f} ms per sample")
        print(f"Baseline inference time: {baseline_time:.4f} ms per sample")
        
        # For very small timing values (< 1ms), use an absolute threshold instead of a percentage
        # This is because tiny variations in timing are expected and not meaningful
        if baseline_time < 1.0:  # If baseline is less than 1ms
            # Allow for small absolute variation (0.5ms) for very fast inference times
            max_allowed = baseline_time + 0.5
            print(f"Using absolute threshold: {max_allowed:.4f} ms (baseline + 0.5ms)")
        else:
            # Allow for larger variation (50% tolerance) for timing tests which can be affected by system load
            max_allowed = baseline_time * 1.5
            print(f"Using relative threshold: {max_allowed:.4f} ms (baseline * 1.5)")
        
        assert avg_time_ms <= max_allowed, (
            f"Inference time has degraded significantly: {avg_time_ms:.4f} ms per sample vs. "
            f"baseline {baseline_time:.4f} ms (max allowed: {max_allowed:.4f} ms)"
        )
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_training_convergence_regression(self, setup):
        """Test that training convergence doesn't degrade."""
        model = setup["model"]
        train_loader = setup["train_loader"]
        baseline = setup["baseline"]
        baseline_path = setup["baseline_path"]
        
        # Train for a few epochs
        device = torch.device("cpu")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            epochs=5,  # Few epochs for quick testing
            lr=1e-3,
            device=device
        )
        
        # Evaluate final loss
        model.eval()
        dataset = setup["dataset"]
        criterion = torch.nn.MSELoss()
        reconstruction_loss = 0.0
        actuator_loss = 0.0
        
        with torch.no_grad():
            for i in range(len(dataset)):
                x_lq, x_hq, y_cmd = dataset[i]
                x_lq = x_lq.unsqueeze(0)
                x_hq = x_hq.unsqueeze(0)
                y_cmd = y_cmd.unsqueeze(0)
                
                reconstructed_hq, act_command, _, _ = model(x_lq, x_hq)
                
                reconstruction_loss += criterion(reconstructed_hq, x_hq).item()
                actuator_loss += criterion(act_command, y_cmd).item()
        
        reconstruction_mse = reconstruction_loss / len(dataset)
        actuator_mse = actuator_loss / len(dataset)
        
        # Get baseline values
        current_metrics = {
            "reconstruction_mse": reconstruction_mse,
            "actuator_mse": actuator_mse
        }
        baseline = self._get_or_create_baseline(current_metrics, baseline_path)
        
        # Check for regression
        baseline_recon_mse = baseline["reconstruction_mse"]
        baseline_act_mse = baseline["actuator_mse"]
        
        # Log information
        print(f"\nTraining Convergence Regression Test:")
        print(f"Current reconstruction MSE: {reconstruction_mse:.4f}")
        print(f"Baseline reconstruction MSE: {baseline_recon_mse:.4f}")
        print(f"Current actuator MSE: {actuator_mse:.4f}")
        print(f"Baseline actuator MSE: {baseline_act_mse:.4f}")
        
        # Allow for some variation (10% tolerance)
        max_recon_allowed = baseline_recon_mse * 1.10
        max_act_allowed = baseline_act_mse * 1.10
        
        assert reconstruction_mse <= max_recon_allowed, (
            f"Reconstruction loss has degraded: {reconstruction_mse:.4f} vs. "
            f"baseline {baseline_recon_mse:.4f} (max allowed: {max_recon_allowed:.4f})"
        )
        
        assert actuator_mse <= max_act_allowed, (
            f"Actuator loss has degraded: {actuator_mse:.4f} vs. "
            f"baseline {baseline_act_mse:.4f} (max allowed: {max_act_allowed:.4f})"
        ) 