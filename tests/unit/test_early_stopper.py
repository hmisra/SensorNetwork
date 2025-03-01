"""
Unit tests for the EarlyStopper component.
"""
import pytest
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor_actuator_network import EarlyStopper, set_seed


class TestEarlyStopper:
    """Tests for the EarlyStopper class."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        set_seed(42)  # For reproducibility
        return {
            "patience": 5,
            "min_delta": 0.01
        }
    
    def test_initialization(self, setup):
        """Test that the EarlyStopper can be initialized correctly."""
        patience = setup["patience"]
        min_delta = setup["min_delta"]
        
        stopper = EarlyStopper(patience=patience, min_delta=min_delta)
        
        assert stopper.patience == patience
        assert stopper.min_delta == min_delta
        assert stopper.counter == 0
        assert stopper.best_loss is None
        assert stopper.should_stop is False
    
    def test_first_loss_always_best(self, setup):
        """Test that the first loss value is always set as the best loss."""
        stopper = EarlyStopper(**setup)
        
        initial_loss = 1.0
        stopper.check(initial_loss)
        
        assert stopper.best_loss == initial_loss
        assert stopper.counter == 0
        assert stopper.should_stop is False
    
    def test_improved_loss_resets_counter(self, setup):
        """Test that an improved loss value resets the counter."""
        stopper = EarlyStopper(**setup)
        
        # Initial loss
        stopper.check(1.0)
        assert stopper.counter == 0
        
        # Worse loss, counter should increment
        stopper.check(1.1)
        assert stopper.counter == 1
        
        # Better loss by more than min_delta, counter should reset
        stopper.check(0.98)  # 1.0 - 0.98 > min_delta (0.01)
        assert stopper.counter == 0
        assert stopper.best_loss == 0.98
    
    def test_marginally_improved_loss_increments_counter(self, setup):
        """Test that a marginally improved loss still increments the counter if below min_delta."""
        stopper = EarlyStopper(**setup)
        min_delta = setup["min_delta"]  # 0.01
        
        # Initial loss
        initial_loss = 1.0
        stopper.check(initial_loss)
        
        # Slightly better loss, but improvement < min_delta, so counter should increment
        stopper.check(initial_loss - (min_delta * 0.5))  # 0.995, which is only 0.005 better
        assert stopper.counter == 1
        assert stopper.best_loss == initial_loss  # Best loss should not update
    
    def test_stops_after_patience_exceeded(self, setup):
        """Test that the stopper signals to stop after patience is exceeded."""
        stopper = EarlyStopper(**setup)
        patience = setup["patience"]  # 5
        
        # Initial loss
        stopper.check(1.0)
        
        # Consistently worse loss, should increment counter each time
        for i in range(patience):
            stopper.check(1.1)
            assert stopper.counter == i + 1
            assert stopper.should_stop == (i + 1 >= patience)
        
        assert stopper.should_stop is True
    
    def test_continues_with_improvements(self, setup):
        """Test that training continues if improvements keep occurring."""
        stopper = EarlyStopper(**setup)
        patience = setup["patience"]  # 5
        
        # Initial loss
        current_loss = 1.0
        stopper.check(current_loss)
        
        # Alternate between worse and better loss
        for i in range(patience * 2):  # Should run longer than patience
            if i % 2 == 0:
                # Worse loss
                stopper.check(current_loss + 0.1)
                assert stopper.counter == 1
            else:
                # Better loss
                current_loss -= 0.02  # Improvement > min_delta (0.01)
                stopper.check(current_loss)
                assert stopper.counter == 0
        
        assert stopper.should_stop is False 