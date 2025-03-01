"""
Unit tests for training utilities including learning rate scheduler.
"""
import pytest
import torch
import torch.optim as optim
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor_actuator_network import SensorAugmentor, set_seed


class TestLearningRateScheduler:
    """Tests for learning rate scheduler behavior."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment with model and optimizer."""
        set_seed(42)  # For reproducibility
        
        # Create a small model for testing
        model = SensorAugmentor(
            sensor_dim=8,
            hidden_dim=16,
            output_dim=1,
            num_resblocks=1
        )
        
        # Initialize optimizer with a specified learning rate
        initial_lr = 0.01
        optimizer = optim.Adam(model.parameters(), lr=initial_lr)
        
        return {
            "model": model,
            "optimizer": optimizer,
            "initial_lr": initial_lr
        }
    
    def test_reduce_lr_on_plateau_scheduler_decreases_lr(self, setup):
        """Test that ReduceLROnPlateau reduces learning rate when loss plateaus."""
        optimizer = setup["optimizer"]
        initial_lr = setup["initial_lr"]
        
        # Create scheduler with small patience for testing
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,  # Half the learning rate
            patience=2,   # After 2 epochs with no improvement
            verbose=False
        )
        
        # Initial learning rate should match what we set
        assert optimizer.param_groups[0]['lr'] == initial_lr
        
        # Simulate training with plateau in loss
        losses = [0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7]
        
        # We should see a learning rate reduction after the plateau
        for loss in losses:
            scheduler.step(loss)
        
        # Learning rate should have decreased
        assert optimizer.param_groups[0]['lr'] < initial_lr
        
        # Given our parameters, it should have reduced to initial_lr * 0.5
        assert optimizer.param_groups[0]['lr'] == pytest.approx(initial_lr * 0.5)
    
    def test_multiple_lr_reductions(self, setup):
        """Test that learning rate can be reduced multiple times when loss continues to plateau."""
        optimizer = setup["optimizer"]
        initial_lr = setup["initial_lr"]
        
        # Create scheduler with small patience for testing
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,
            patience=1,  # More aggressive for testing
            verbose=False
        )
        
        # Simulate multiple plateaus
        sequences = [
            [0.9, 0.8],  # Improvement
            [0.8, 0.8, 0.8],  # First plateau: expect LR reduction
            [0.79, 0.79, 0.79],  # Second plateau: expect LR reduction
        ]
        
        expected_lr = initial_lr
        
        for sequence in sequences:
            for loss in sequence:
                scheduler.step(loss)
            
            if len(sequence) > 2:  # If this was a plateau
                expected_lr *= 0.5
                assert optimizer.param_groups[0]['lr'] == pytest.approx(expected_lr)
    
    def test_lr_scheduler_stops_at_min_lr(self, setup):
        """Test that learning rate does not go below min_lr."""
        optimizer = setup["optimizer"]
        initial_lr = setup["initial_lr"]
        
        min_lr = 0.001  # Set a minimum learning rate
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1,  # Aggressive reduction
            patience=1,
            min_lr=min_lr,
            verbose=False
        )
        
        # Simulate multiple plateaus to force reduction below min_lr
        for i in range(10):  # Many plateaus
            scheduler.step(0.5)  # Constant loss
        
        # LR should not drop below min_lr
        assert optimizer.param_groups[0]['lr'] >= min_lr 