"""
Unit tests for the ResidualBlock component.
"""
import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor_actuator_network import ResidualBlock, set_seed


class TestResidualBlock:
    """Tests for the ResidualBlock class."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        set_seed(42)  # For reproducibility
        return {
            "input_dim": 64,
            "batch_size": 16
        }
    
    def test_initialization(self, setup):
        """Test that the ResidualBlock can be initialized correctly."""
        input_dim = setup["input_dim"]
        block = ResidualBlock(input_dim)
        
        assert isinstance(block, nn.Module)
        assert isinstance(block.linear1, nn.Linear)
        assert isinstance(block.linear2, nn.Linear)
        assert block.linear1.in_features == input_dim
        assert block.linear1.out_features == input_dim
        assert block.linear2.in_features == input_dim
        assert block.linear2.out_features == input_dim
    
    def test_forward_pass(self, setup):
        """Test that the forward pass works and maintains the input dimensions."""
        input_dim = setup["input_dim"]
        batch_size = setup["batch_size"]
        
        block = ResidualBlock(input_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = block(x)
        
        assert output.shape == (batch_size, input_dim)
        assert not torch.equal(output, x)  # Output should be different from input
    
    def test_residual_connection(self, setup):
        """Test that the residual connection is working properly."""
        input_dim = setup["input_dim"]
        batch_size = setup["batch_size"]
        
        # Create a special ResidualBlock where linear layers do nothing
        block = ResidualBlock(input_dim)
        # Initialize weights to zero and bias to zero
        with torch.no_grad():
            nn.init.zeros_(block.linear1.weight)
            nn.init.zeros_(block.linear1.bias)
            nn.init.zeros_(block.linear2.weight)
            nn.init.zeros_(block.linear2.bias)
        
        x = torch.randn(batch_size, input_dim)
        output = block(x)
        
        # With zero weights and bias, the output should be equal to ReLU(x)
        # since the residual connection adds the input to the zero result
        expected = nn.functional.relu(x)
        assert torch.allclose(output, expected) 