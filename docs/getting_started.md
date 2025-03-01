# Getting Started with SensorAugmentor

This guide will help you get started with the SensorAugmentor framework for enhancing low-quality sensor data and generating actuator commands.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SensorAugmentor.git
   cd SensorAugmentor
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### Training a Model with Synthetic Data

The simplest way to get started is to use the built-in synthetic dataset:

```python
import torch
from sensor_actuator_network import SensorAugmentor, SyntheticSensorDataset, train_model, set_seed

# Set seed for reproducibility
set_seed(42)

# Parameters
sensor_dim = 32     # Dimension of sensor data
hidden_dim = 64     # Dimension of hidden layers
output_dim = 1      # Dimension of actuator command
batch_size = 32     # Batch size for training
epochs = 20         # Number of training epochs

# Create synthetic dataset
dataset = SyntheticSensorDataset(num_samples=2000, sensor_dim=sensor_dim, noise_factor=0.3)

# Split into train/val sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

g = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=g)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create model
model = SensorAugmentor(
    sensor_dim=sensor_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_resblocks=2
)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train model
train_model(model, train_loader, val_loader, epochs=epochs, lr=1e-3, device=device)

# Save model
torch.save(model.state_dict(), "sensor_augmentor_model.pth")
```

### Using a Trained Model for Inference

After training, you can use the model for inference:

```python
# Load trained model
model = SensorAugmentor(sensor_dim=sensor_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model.load_state_dict(torch.load("sensor_augmentor_model.pth"))
model.eval()
model.to(device)

# Example inference
# Create a low-quality sensor reading
lq_sensor_data = torch.randn(1, sensor_dim)

# Normalize using dataset statistics
lq_sensor_data_norm = (lq_sensor_data - dataset.mean_lq) / dataset.std_lq
lq_sensor_data_norm = lq_sensor_data_norm.to(device)

# Get predictions
with torch.no_grad():
    reconstructed_hq, actuator_command, _, _ = model(lq_sensor_data_norm)

# Denormalize reconstructed high-quality signal
reconstructed_hq_denorm = reconstructed_hq.cpu() * dataset.std_hq + dataset.mean_hq

print("Reconstructed HQ signal:", reconstructed_hq_denorm)
print("Actuator command:", actuator_command.cpu())
```

### Creating a Custom Dataset

For your own application, you'll need to create a custom dataset. Here's a template:

```python
import torch
import torch.utils.data

class CustomSensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        # Load your data
        # self.data = ...
        
        # Extract LQ and HQ sensor readings
        # self.x_lq = ...
        # self.x_hq = ...
        
        # Calculate actuator commands
        # self.y_cmd = ...
        
        # Calculate normalization statistics
        self.mean_lq = self.x_lq.mean(dim=0, keepdim=True)
        self.std_lq = self.x_lq.std(dim=0, keepdim=True) + 1e-6
        self.mean_hq = self.x_hq.mean(dim=0, keepdim=True)
        self.std_hq = self.x_hq.std(dim=0, keepdim=True) + 1e-6
        
        # Normalize data
        self.x_lq_norm = (self.x_lq - self.mean_lq) / self.std_lq
        self.x_hq_norm = (self.x_hq - self.mean_hq) / self.std_hq
        
        self.transform = transform
    
    def __len__(self):
        return len(self.x_lq)
    
    def __getitem__(self, idx):
        lq = self.x_lq_norm[idx]
        hq = self.x_hq_norm[idx]
        cmd = self.y_cmd[idx]
        
        if self.transform:
            lq = self.transform(lq)
            hq = self.transform(hq)
        
        return lq, hq, cmd
```

Check the `examples/custom_dataset_example.py` file for a more complete example.

## Running Tests

To verify that everything is working correctly, you can run the test suite:

```bash
python run_tests.py
```

This will run both unit and integration tests. To run only specific test types:

```bash
python run_tests.py unit       # Run only unit tests
python run_tests.py integration # Run only integration tests
```

## Next Steps

- Check out the examples directory for more detailed examples
- Read the API documentation for more information on the available classes and functions
- See the environmental sensor example for a practical application 