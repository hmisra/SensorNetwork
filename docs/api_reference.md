# SensorAugmentor API Reference

This document provides detailed information about the classes and functions available in the SensorAugmentor framework.

## Core Classes

### ResidualBlock

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        # ...
    
    def forward(self, x):
        # ...
```

A residual block with two linear layers and a skip connection.

**Parameters:**
- `dim` (int): Dimension of input and output features.

**Methods:**
- `forward(x)`: Forward pass through the residual block.
  - `x` (torch.Tensor): Input tensor of shape (batch_size, dim).
  - Returns: Output tensor of shape (batch_size, dim).

### SensorAugmentor

```python
class SensorAugmentor(nn.Module):
    def __init__(self, sensor_dim=32, hidden_dim=64, output_dim=16, num_resblocks=2):
        # ...
    
    def forward(self, x_lq, x_hq=None):
        # ...
```

Main model for enhancing low-quality sensor data and generating actuator commands.

**Parameters:**
- `sensor_dim` (int): Dimension of the sensor data.
- `hidden_dim` (int): Dimension of the hidden layers and latent representation.
- `output_dim` (int): Dimension of the actuator command.
- `num_resblocks` (int): Number of residual blocks in the encoder.

**Methods:**
- `forward(x_lq, x_hq=None)`: Forward pass through the model.
  - `x_lq` (torch.Tensor): Low-quality sensor data of shape (batch_size, sensor_dim).
  - `x_hq` (torch.Tensor, optional): High-quality sensor data of shape (batch_size, sensor_dim). Required for training, not needed for inference.
  - Returns:
    - `reconstructed_hq` (torch.Tensor): Reconstructed high-quality sensor data.
    - `act_command` (torch.Tensor): Predicted actuator command.
    - `encoded_lq` (torch.Tensor): Latent representation of low-quality data.
    - `encoded_hq` (torch.Tensor or None): Latent representation of high-quality data (if provided).

### SyntheticSensorDataset

```python
class SyntheticSensorDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, sensor_dim=32, noise_factor=0.3):
        # ...
    
    def __len__(self):
        # ...
    
    def __getitem__(self, idx):
        # ...
```

Dataset for generating synthetic sensor data for training and evaluation.

**Parameters:**
- `num_samples` (int): Number of samples to generate.
- `sensor_dim` (int): Dimension of the sensor data.
- `noise_factor` (float): Amount of noise to add to the high-quality data to create low-quality data.

**Attributes:**
- `x_lq` (torch.Tensor): Normalized low-quality sensor data.
- `x_hq` (torch.Tensor): Normalized high-quality sensor data.
- `y_cmd` (torch.Tensor): Ground truth actuator commands.
- `mean_lq`, `std_lq`, `mean_hq`, `std_hq` (torch.Tensor): Normalization statistics.

**Methods:**
- `__len__()`: Returns the number of samples in the dataset.
- `__getitem__(idx)`: Returns the i-th sample.
  - `idx` (int): Index of the sample.
  - Returns: Tuple of (x_lq, x_hq, y_cmd) tensors for the requested index.

### EarlyStopper

```python
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        # ...
    
    def check(self, val_loss):
        # ...
```

Utility class for early stopping during training.

**Parameters:**
- `patience` (int): Number of epochs with no improvement after which training will be stopped.
- `min_delta` (float): Minimum change in validation loss to qualify as an improvement.

**Attributes:**
- `counter` (int): Counter for epochs with no improvement.
- `best_loss` (float or None): Best validation loss observed.
- `should_stop` (bool): Whether training should stop.

**Methods:**
- `check(val_loss)`: Check if training should stop based on the current validation loss.
  - `val_loss` (float): Current validation loss.

## Functions

### set_seed

```python
def set_seed(seed=42):
    # ...
```

Set random seeds for reproducibility.

**Parameters:**
- `seed` (int): Random seed to use.

### train_model

```python
def train_model(model, train_loader, val_loader=None, epochs=30, lr=1e-3, device="cpu"):
    # ...
```

Train the SensorAugmentor model.

**Parameters:**
- `model` (SensorAugmentor): Model to train.
- `train_loader` (torch.utils.data.DataLoader): DataLoader for training data.
- `val_loader` (torch.utils.data.DataLoader, optional): DataLoader for validation data.
- `epochs` (int): Number of training epochs.
- `lr` (float): Learning rate for the optimizer.
- `device` (str): Device to use for training ('cpu' or 'cuda').

## Usage Examples

### Basic Training and Inference

```python
# Initialize model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)

# Create dataset and loaders
dataset = SyntheticSensorDataset(num_samples=2000, sensor_dim=32)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Train model
train_model(model, train_loader, epochs=20)

# Inference
model.eval()
with torch.no_grad():
    lq_input = torch.randn(1, 32)  # Example input
    hq_reconstruction, actuator_cmd, _, _ = model(lq_input)
```

### Custom Training Loop

If you need more control over the training process:

```python
# Initialize model and optimizer
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(10):
    for x_lq, x_hq, y_cmd in train_loader:
        # Forward pass
        recon_hq, act_cmd, encoded_lq, encoded_hq = model(x_lq, x_hq)
        
        # Calculate losses
        loss_recon = criterion(recon_hq, x_hq)
        loss_cmd = criterion(act_cmd, y_cmd)
        loss_latent = criterion(encoded_lq, encoded_hq)
        
        # Combined loss
        loss = loss_recon + loss_cmd + 0.1 * loss_latent
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
``` 