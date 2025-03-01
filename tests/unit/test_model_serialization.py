"""
Unit tests for model serialization and loading.
"""
import pytest
import torch
import os
import tempfile
import sys
from pathlib import Path

# Add the root directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sensor_actuator_network import SensorAugmentor, set_seed, SyntheticSensorDataset


class TestModelSerialization:
    """Tests for model saving and loading functionality."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        set_seed(42)  # For reproducibility
        
        # Create model and data
        model = SensorAugmentor(
            sensor_dim=16,
            hidden_dim=32,
            output_dim=4,
            num_resblocks=2
        )
        
        # Create sample data for inference testing
        x_lq = torch.randn(8, 16)
        
        # Create temporary directory for model files
        temp_dir = tempfile.mkdtemp()
        
        return {
            "model": model,
            "x_lq": x_lq,
            "temp_dir": temp_dir
        }
    
    def teardown_method(self, method):
        """Clean up temporary files after each test."""
        # If there's a temp_dir in the test's namespace, clean it up
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
    
    def test_save_and_load_model(self, setup):
        """Test saving and loading a model with torch.save and torch.load."""
        model = setup["model"]
        x_lq = setup["x_lq"]
        temp_dir = setup["temp_dir"]
        
        # Get model predictions before saving
        model.eval()
        with torch.no_grad():
            before_save_outputs = model(x_lq)
        
        # Save the model
        model_path = os.path.join(temp_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Check that the file exists
        assert os.path.exists(model_path)
        
        # Create a new model with the same architecture
        new_model = SensorAugmentor(
            sensor_dim=16,
            hidden_dim=32,
            output_dim=4,
            num_resblocks=2
        )
        
        # Load the saved state
        new_model.load_state_dict(torch.load(model_path))
        new_model.eval()
        
        # Get predictions from the loaded model
        with torch.no_grad():
            after_load_outputs = new_model(x_lq)
        
        # Check that predictions are the same
        # We only care about the first two outputs (reconstructed_hq and act_command)
        assert torch.allclose(before_save_outputs[0], after_load_outputs[0])
        assert torch.allclose(before_save_outputs[1], after_load_outputs[1])
    
    def test_save_load_with_metadata(self, setup):
        """Test saving and loading a model with metadata."""
        model = setup["model"]
        temp_dir = setup["temp_dir"]
        
        # Create metadata dictionary
        metadata = {
            "model_version": "1.0.0",
            "training_date": "2023-06-15",
            "sensor_dim": 16,
            "hidden_dim": 32,
            "output_dim": 4,
            "num_resblocks": 2,
            "train_loss": 0.123,
            "val_loss": 0.234
        }
        
        # Save the model with metadata
        model_path = os.path.join(temp_dir, "model_with_metadata.pt")
        save_dict = {
            "state_dict": model.state_dict(),
            "metadata": metadata
        }
        torch.save(save_dict, model_path)
        
        # Check that the file exists
        assert os.path.exists(model_path)
        
        # Load the saved file
        loaded_dict = torch.load(model_path)
        
        # Check that metadata was preserved
        assert "metadata" in loaded_dict
        assert loaded_dict["metadata"] == metadata
        
        # Load the state dict into a new model
        new_model = SensorAugmentor(
            sensor_dim=metadata["sensor_dim"],
            hidden_dim=metadata["hidden_dim"],
            output_dim=metadata["output_dim"],
            num_resblocks=metadata["num_resblocks"]
        )
        new_model.load_state_dict(loaded_dict["state_dict"])
    
    def test_save_load_with_dataset_params(self, setup):
        """Test saving and loading a model with dataset parameters."""
        model = setup["model"]
        temp_dir = setup["temp_dir"]
        
        # Create dataset with parameters we want to save
        dataset = SyntheticSensorDataset(
            num_samples=100,
            sensor_dim=16,
            noise_factor=0.25
        )
        
        # Create metadata with dataset normalization parameters
        metadata = {
            "sensor_dim": 16,
            "hidden_dim": 32,
            "output_dim": 4,
            "dataset_stats": {
                "mean_lq": dataset.mean_lq.numpy().tolist(),
                "std_lq": dataset.std_lq.numpy().tolist(),
                "mean_hq": dataset.mean_hq.numpy().tolist(),
                "std_hq": dataset.std_hq.numpy().tolist()
            }
        }
        
        # Save the model with dataset parameters
        model_path = os.path.join(temp_dir, "model_with_dataset.pt")
        save_dict = {
            "state_dict": model.state_dict(),
            "metadata": metadata
        }
        torch.save(save_dict, model_path)
        
        # Load the saved file
        loaded_dict = torch.load(model_path)
        
        # Check that dataset stats were preserved
        assert "dataset_stats" in loaded_dict["metadata"]
        dataset_stats = loaded_dict["metadata"]["dataset_stats"]
        
        # Verify we can reconstruct tensors from the loaded data
        loaded_mean_lq = torch.tensor(dataset_stats["mean_lq"])
        loaded_std_lq = torch.tensor(dataset_stats["std_lq"])
        
        # Shape should match original
        assert loaded_mean_lq.shape == dataset.mean_lq.shape
        assert loaded_std_lq.shape == dataset.std_lq.shape
        
        # Values should match original
        assert torch.allclose(loaded_mean_lq, dataset.mean_lq)
        assert torch.allclose(loaded_std_lq, dataset.std_lq)
    
    def test_checkpointing_during_training(self, setup):
        """Test saving checkpoints during training."""
        model = setup["model"]
        temp_dir = setup["temp_dir"]
        
        # Create a simple optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        checkpoint = {
            "epoch": 10,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 0.123
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Check that the file exists
        assert os.path.exists(checkpoint_path)
        
        # Create new model and optimizer
        new_model = SensorAugmentor(
            sensor_dim=16,
            hidden_dim=32,
            output_dim=4,
            num_resblocks=2
        )
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.002)  # Different learning rate
        
        # Load the checkpoint
        loaded_checkpoint = torch.load(checkpoint_path)
        new_model.load_state_dict(loaded_checkpoint["model_state_dict"])
        new_optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        epoch = loaded_checkpoint["epoch"]
        loss = loaded_checkpoint["loss"]
        
        # Check that everything was restored correctly
        assert epoch == 10
        assert loss == 0.123
        
        # Most importantly, the optimizer's learning rate should be restored
        assert new_optimizer.param_groups[0]["lr"] == 0.001  # Original learning rate 