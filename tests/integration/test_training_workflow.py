"""
Integration tests for the SensorAugmentor training workflow.
"""
import pytest
import torch
import torch.utils.data
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


class TestTrainingWorkflow:
    """Integration tests for the training workflow."""
    
    @pytest.fixture
    def setup(self):
        """Set up test environment."""
        set_seed(42)  # For reproducibility
        
        # Small dataset and model for quick testing
        params = {
            "sensor_dim": 16,
            "hidden_dim": 32,
            "output_dim": 1,
            "num_samples": 200,
            "batch_size": 16,
            "num_resblocks": 1,
            "epochs": 3
        }
        
        # Create dataset
        dataset = SyntheticSensorDataset(
            num_samples=params["num_samples"], 
            sensor_dim=params["sensor_dim"]
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        g = torch.Generator()
        g.manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=g
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=params["batch_size"], 
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=params["batch_size"], 
            shuffle=False
        )
        
        # Create model
        model = SensorAugmentor(
            sensor_dim=params["sensor_dim"],
            hidden_dim=params["hidden_dim"],
            output_dim=params["output_dim"],
            num_resblocks=params["num_resblocks"]
        )
        
        return {
            "params": params,
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "dataset": dataset
        }
    
    def test_training_improves_loss(self, setup):
        """Test that training reduces the loss."""
        model = setup["model"]
        train_loader = setup["train_loader"]
        val_loader = setup["val_loader"]
        params = setup["params"]
        
        # Get initial loss
        model.eval()
        criterion = torch.nn.MSELoss()
        initial_loss = 0.0
        
        with torch.no_grad():
            for x_lq, x_hq, y_cmd in val_loader:
                pred_hq, pred_act, _, _ = model(x_lq, x_hq)
                loss = criterion(pred_hq, x_hq) + criterion(pred_act, y_cmd)
                initial_loss += loss.item()
        
        initial_loss /= len(val_loader)
        
        # Train model for a few epochs
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=params["epochs"],
            lr=1e-3,
            device="cpu"
        )
        
        # Get final loss
        model.eval()
        final_loss = 0.0
        
        with torch.no_grad():
            for x_lq, x_hq, y_cmd in val_loader:
                pred_hq, pred_act, _, _ = model(x_lq, x_hq)
                loss = criterion(pred_hq, x_hq) + criterion(pred_act, y_cmd)
                final_loss += loss.item()
        
        final_loss /= len(val_loader)
        
        # Assert that training improved the loss
        assert final_loss < initial_loss, (
            f"Training did not improve loss. Initial: {initial_loss}, Final: {final_loss}"
        )
    
    def test_inference_workflow(self, setup):
        """Test the inference workflow with a trained model."""
        model = setup["model"]
        dataset = setup["dataset"]
        params = setup["params"]
        
        # Train the model minimally
        train_loader = setup["train_loader"]
        val_loader = setup["val_loader"]
        train_model(model, train_loader, val_loader, epochs=1, lr=1e-3, device="cpu")
        
        # Run inference on a random sample
        model.eval()
        
        # Create a test input
        test_input = torch.randn(1, params["sensor_dim"])
        
        # Normalize using dataset stats
        test_input_norm = (test_input - dataset.mean_lq) / dataset.std_lq
        
        # Get predictions
        with torch.no_grad():
            reconstructed_hq, act_command, _, _ = model(test_input_norm)
        
        # Check output dimensions
        assert reconstructed_hq.shape == (1, params["sensor_dim"])
        assert act_command.shape == (1, params["output_dim"])
        
        # Verify we can denormalize the output
        denorm_hq = reconstructed_hq * dataset.std_hq + dataset.mean_hq
        assert denorm_hq.shape == (1, params["sensor_dim"]) 