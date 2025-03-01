"""
Unit tests for the SensorAugmentor model.
"""
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor_actuator_network import SensorAugmentor, set_seed


class TestSensorAugmentor:
    """Tests for the SensorAugmentor class."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        set_seed(42)  # For reproducibility
        return {
            "sensor_dim": 32,
            "hidden_dim": 64,
            "output_dim": 8,
            "num_resblocks": 2,
            "batch_size": 16
        }
    
    def test_initialization(self, setup):
        """Test that the SensorAugmentor can be initialized correctly."""
        sensor_dim = setup["sensor_dim"]
        hidden_dim = setup["hidden_dim"]
        output_dim = setup["output_dim"]
        num_resblocks = setup["num_resblocks"]
        
        model = SensorAugmentor(
            sensor_dim=sensor_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_resblocks=num_resblocks
        )
        
        assert isinstance(model, nn.Module)
        assert isinstance(model.encoder, nn.Sequential)
        assert isinstance(model.hq_reconstructor, nn.Sequential)
        assert isinstance(model.actuator_head, nn.Linear)
        assert isinstance(model.post_encoding_resblock, nn.Module)
        
        # Check dimensions
        assert model.actuator_head.in_features == hidden_dim
        assert model.actuator_head.out_features == output_dim
    
    def test_forward_pass_with_hq(self, setup):
        """Test forward pass with both LQ and HQ inputs."""
        sensor_dim = setup["sensor_dim"]
        hidden_dim = setup["hidden_dim"]
        output_dim = setup["output_dim"]
        batch_size = setup["batch_size"]
        
        model = SensorAugmentor(
            sensor_dim=sensor_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        x_lq = torch.randn(batch_size, sensor_dim)
        x_hq = torch.randn(batch_size, sensor_dim)
        
        reconstructed_hq, act_command, encoded_lq, encoded_hq = model(x_lq, x_hq)
        
        # Check output shapes
        assert reconstructed_hq.shape == (batch_size, sensor_dim)
        assert act_command.shape == (batch_size, output_dim)
        assert encoded_lq.shape == (batch_size, hidden_dim)
        assert encoded_hq.shape == (batch_size, hidden_dim)
    
    def test_forward_pass_without_hq(self, setup):
        """Test forward pass with only LQ input."""
        sensor_dim = setup["sensor_dim"]
        hidden_dim = setup["hidden_dim"]
        output_dim = setup["output_dim"]
        batch_size = setup["batch_size"]
        
        model = SensorAugmentor(
            sensor_dim=sensor_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        x_lq = torch.randn(batch_size, sensor_dim)
        
        reconstructed_hq, act_command, encoded_lq, encoded_hq = model(x_lq)
        
        # Check output shapes
        assert reconstructed_hq.shape == (batch_size, sensor_dim)
        assert act_command.shape == (batch_size, output_dim)
        assert encoded_lq.shape == (batch_size, hidden_dim)
        assert encoded_hq is None  # Should be None when x_hq is not provided
    
    def test_encoder_shared_weights(self, setup):
        """Test that encoder weights are shared between LQ and HQ processing."""
        sensor_dim = setup["sensor_dim"]
        hidden_dim = setup["hidden_dim"]
        output_dim = setup["output_dim"]
        batch_size = setup["batch_size"]
        
        model = SensorAugmentor(
            sensor_dim=sensor_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        # Create identical LQ and HQ inputs
        identical_input = torch.randn(batch_size, sensor_dim)
        
        _, _, encoded_lq, encoded_hq = model(identical_input, identical_input)
        
        # Since inputs are identical and weights are shared, encodings should be identical
        assert torch.allclose(encoded_lq, encoded_hq) 