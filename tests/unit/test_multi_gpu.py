"""
Unit tests for multi-GPU support.
These tests will be skipped if multiple GPUs are not available.
"""
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor_actuator_network import SensorAugmentor, set_seed


# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Testing multi-GPU support requires at least 2 GPUs"
)


class TestMultiGPUSupport:
    """Tests for multi-GPU support using DataParallel."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        set_seed(42)  # For reproducibility
        
        # Create model and sample data
        model = SensorAugmentor(
            sensor_dim=32,
            hidden_dim=64,
            output_dim=8
        )
        
        batch_size = 16
        x_lq = torch.randn(batch_size, 32)
        x_hq = torch.randn(batch_size, 32)
        
        return {
            "model": model,
            "x_lq": x_lq,
            "x_hq": x_hq,
            "batch_size": batch_size
        }
    
    def test_dataparallel_initialization(self, setup):
        """Test that the model can be wrapped in DataParallel."""
        model = setup["model"]
        
        # Wrap model in DataParallel
        parallel_model = nn.DataParallel(model)
        
        # Model should now be a DataParallel instance
        assert isinstance(parallel_model, nn.DataParallel)
        assert parallel_model.module is model  # Original model should be accessible via .module
    
    def test_dataparallel_forward_pass(self, setup):
        """Test that the model can perform a forward pass when wrapped in DataParallel."""
        model = setup["model"]
        x_lq = setup["x_lq"]
        x_hq = setup["x_hq"]
        batch_size = setup["batch_size"]
        
        # Move everything to CUDA
        model = model.cuda()
        x_lq = x_lq.cuda()
        x_hq = x_hq.cuda()
        
        # Wrap model in DataParallel
        parallel_model = nn.DataParallel(model)
        
        # Perform forward pass
        reconstructed_hq, act_command, encoded_lq, encoded_hq = parallel_model(x_lq, x_hq)
        
        # Verify output shapes
        assert reconstructed_hq.shape == (batch_size, 32)
        assert act_command.shape == (batch_size, 8)
        assert encoded_lq.shape == (batch_size, 64)
        assert encoded_hq.shape == (batch_size, 64)
        
        # Verify outputs are on the same device as inputs
        assert reconstructed_hq.device == x_lq.device
    
    def test_dataparallel_backward_pass(self, setup):
        """Test that the model can perform a backward pass when wrapped in DataParallel."""
        model = setup["model"]
        x_lq = setup["x_lq"]
        x_hq = setup["x_hq"]
        
        # Move everything to CUDA
        model = model.cuda()
        x_lq = x_lq.cuda()
        x_hq = x_hq.cuda()
        
        # Target for loss calculation
        target_act = torch.randn_like(torch.zeros(x_lq.size(0), 8)).cuda()
        
        # Wrap model in DataParallel
        parallel_model = nn.DataParallel(model)
        
        # Forward pass
        reconstructed_hq, act_command, encoded_lq, encoded_hq = parallel_model(x_lq, x_hq)
        
        # Calculate loss
        loss = nn.MSELoss()(reconstructed_hq, x_hq) + nn.MSELoss()(act_command, target_act)
        
        # Check that we can backpropagate
        loss.backward()
        
        # All parameters should have gradients now
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
    
    def test_large_batch_distribution(self, setup):
        """Test that large batches are distributed across GPUs."""
        model = setup["model"].cuda()
        
        # Create a very large batch
        large_batch_size = 128
        x_large = torch.randn(large_batch_size, 32).cuda()
        
        # Wrap model in DataParallel
        parallel_model = nn.DataParallel(model)
        
        # Forward pass with large batch (no HQ input for simplicity)
        reconstructed_hq, act_command, encoded_lq, _ = parallel_model(x_large)
        
        # Output should match the large batch size
        assert reconstructed_hq.shape == (large_batch_size, 32)
        assert act_command.shape == (large_batch_size, 8)
        assert encoded_lq.shape == (large_batch_size, 64) 