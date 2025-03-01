# SensorAugmentor API Reference

Welcome to the SensorAugmentor API reference documentation. This document provides detailed information about all the classes, methods, and functions available in the SensorAugmentor framework.

## Table of Contents

- [Core Components](#core-components)
  - [ResidualBlock](#residualblock)
  - [SensorAugmentor](#sensoraugmentor)
  - [SyntheticSensorDataset](#syntheticdataset)
  - [EarlyStopper](#earlystopper)
- [Training Functions](#training-functions)
  - [train_model](#train_model)
  - [set_seed](#set_seed)
- [Advanced Components](#advanced-components)
  - [ModelSerializer](#modelserializer)
  - [DataNormalizer](#datanormalizer)
  - [UncertaintyEstimator](#uncertaintyestimator)
- [Utility Functions](#utility-functions)
  - [visualize_reconstruction](#visualize_reconstruction)
  - [evaluate_model](#evaluate_model)
  - [calculate_metrics](#calculate_metrics)

## Core Components

### ResidualBlock

```python
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, activation: nn.Module = nn.ReLU()):
        # ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ...
```

A residual block with two linear layers and a skip connection.

**Parameters:**
- `dim` (int): Dimension of input and output features.
- `activation` (nn.Module, optional): Activation function to use. Default: `nn.ReLU()`.

**Attributes:**
- `linear1` (nn.Linear): First linear layer with shape (dim, dim).
- `linear2` (nn.Linear): Second linear layer with shape (dim, dim).
- `activation` (nn.Module): Activation function used between layers.

**Methods:**
- `forward(x)`: Forward pass through the residual block.
  - **Parameters:**
    - `x` (torch.Tensor): Input tensor of shape (batch_size, dim).
  - **Returns:**
    - torch.Tensor: Output tensor of shape (batch_size, dim).

**Example:**
```python
import torch
from sensor_actuator_network import ResidualBlock

# Create a residual block with 64 dimensions
block = ResidualBlock(64)

# Create a random input tensor
x = torch.randn(32, 64)  # batch_size=32, dim=64

# Forward pass
output = block(x)
print(output.shape)  # torch.Size([32, 64])
```

### SensorAugmentor

```python
class SensorAugmentor(nn.Module):
    def __init__(self, 
                 sensor_dim: int = 32, 
                 hidden_dim: int = 64, 
                 output_dim: int = 16, 
                 num_resblocks: int = 2,
                 activation: nn.Module = nn.ReLU()):
        # ...
    
    def forward(self, x_lq: torch.Tensor, x_hq: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        # ...
```

Main model for enhancing low-quality sensor data and generating actuator commands. Uses a teacher-student architecture with residual connections.

**Parameters:**
- `sensor_dim` (int, optional): Dimension of the sensor data. Default: 32.
- `hidden_dim` (int, optional): Dimension of the hidden layers and latent representation. Default: 64.
- `output_dim` (int, optional): Dimension of the actuator command. Default: 16.
- `num_resblocks` (int, optional): Number of residual blocks in the encoder. Default: 2.
- `activation` (nn.Module, optional): Activation function to use. Default: `nn.ReLU()`.

**Attributes:**
- `encoder` (nn.Sequential): Encoder network that transforms sensor data into latent representations.
- `hq_reconstructor` (nn.Sequential): Network that reconstructs high-quality sensor data from latent representations.
- `actuator_head` (nn.Linear): Linear layer that maps latent representations to actuator commands.
- `post_encoding_resblock` (ResidualBlock): Additional residual block after encoding.

**Methods:**
- `forward(x_lq, x_hq=None)`: Forward pass through the model.
  - **Parameters:**
    - `x_lq` (torch.Tensor): Low-quality sensor data of shape (batch_size, sensor_dim).
    - `x_hq` (torch.Tensor, optional): High-quality sensor data of shape (batch_size, sensor_dim). Required during training for teacher-student learning, not needed for inference.
  - **Returns:**
    - `reconstructed_hq` (torch.Tensor): Reconstructed high-quality sensor data.
    - `act_command` (torch.Tensor): Predicted actuator command.
    - `encoded_lq` (torch.Tensor): Latent representation of low-quality data.
    - `encoded_hq` (torch.Tensor or None): Latent representation of high-quality data (if provided).

**Example:**
```python
import torch
from sensor_actuator_network import SensorAugmentor

# Create model
model = SensorAugmentor(
    sensor_dim=32,
    hidden_dim=64,
    output_dim=1,
    num_resblocks=3
)

# Training mode with both LQ and HQ data
batch_size = 16
x_lq = torch.randn(batch_size, 32)  # Low-quality sensor data
x_hq = torch.randn(batch_size, 32)  # High-quality sensor data

reconstructed, actuator_cmd, latent_lq, latent_hq = model(x_lq, x_hq)

print(f"Reconstructed shape: {reconstructed.shape}")  # torch.Size([16, 32])
print(f"Actuator command shape: {actuator_cmd.shape}")  # torch.Size([16, 1])
print(f"Latent LQ shape: {latent_lq.shape}")  # torch.Size([16, 64])
print(f"Latent HQ shape: {latent_hq.shape}")  # torch.Size([16, 64])

# Inference mode with only LQ data
model.eval()
with torch.no_grad():
    x_lq_test = torch.randn(1, 32)  # Single sample
    reconstructed, actuator_cmd, latent, _ = model(x_lq_test)
    print(f"Inference reconstructed shape: {reconstructed.shape}")  # torch.Size([1, 32])
    print(f"Inference actuator command: {actuator_cmd}")
```

### SyntheticSensorDataset

```python
class SyntheticSensorDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 num_samples: int = 1000, 
                 sensor_dim: int = 32, 
                 noise_factor: float = 0.3,
                 seed: Optional[int] = None):
        # ...
    
    def __len__(self) -> int:
        # ...
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ...
```

Dataset for generating synthetic sensor data for training and evaluation.

**Parameters:**
- `num_samples` (int, optional): Number of samples to generate. Default: 1000.
- `sensor_dim` (int, optional): Dimension of the sensor data. Default: 32.
- `noise_factor` (float, optional): Amount of noise to add to the high-quality data to create low-quality data. Default: 0.3.
- `seed` (int, optional): Random seed for reproducibility. Default: None.

**Attributes:**
- `x_lq` (torch.Tensor): Normalized low-quality sensor data with shape (num_samples, sensor_dim).
- `x_hq` (torch.Tensor): Normalized high-quality sensor data with shape (num_samples, sensor_dim).
- `y_cmd` (torch.Tensor): Ground truth actuator commands with shape (num_samples, output_dim).
- `mean_lq`, `std_lq`, `mean_hq`, `std_hq` (torch.Tensor): Normalization statistics for the sensor data.

**Methods:**
- `__len__()`: Returns the number of samples in the dataset.
  - **Returns:**
    - int: Number of samples.
- `__getitem__(idx)`: Returns the idx-th sample.
  - **Parameters:**
    - `idx` (int): Index of the sample.
  - **Returns:**
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of (x_lq, x_hq, y_cmd) tensors for the requested index.

**Example:**
```python
import torch
from sensor_actuator_network import SyntheticSensorDataset
from torch.utils.data import DataLoader, random_split

# Create dataset with reproducible seed
dataset = SyntheticSensorDataset(
    num_samples=2000,
    sensor_dim=32,
    noise_factor=0.3,
    seed=42
)

# Access a sample
x_lq, x_hq, y_cmd = dataset[0]
print(f"LQ sensor shape: {x_lq.shape}")  # torch.Size([32])
print(f"HQ sensor shape: {x_hq.shape}")  # torch.Size([32])
print(f"Actuator command shape: {y_cmd.shape}")  # torch.Size([1])

# Split into train/validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

g = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Retrieve normalization statistics for future use
print(f"LQ mean: {dataset.mean_lq}")
print(f"LQ std: {dataset.std_lq}")
print(f"HQ mean: {dataset.mean_hq}")
print(f"HQ std: {dataset.std_hq}")
```

### EarlyStopper

```python
class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        # ...
    
    def check(self, val_loss: float) -> bool:
        # ...
```

Utility class for early stopping during training.

**Parameters:**
- `patience` (int, optional): Number of epochs with no improvement after which training will be stopped. Default: 5.
- `min_delta` (float, optional): Minimum change in validation loss to qualify as an improvement. Default: 0.0.

**Attributes:**
- `patience` (int): Number of epochs to wait for improvement.
- `min_delta` (float): Minimum improvement required to reset counter.
- `counter` (int): Counter for epochs with no improvement.
- `best_loss` (float or None): Best validation loss observed.
- `should_stop` (bool): Whether training should stop.

**Methods:**
- `check(val_loss)`: Check if training should stop based on the current validation loss.
  - **Parameters:**
    - `val_loss` (float): Current validation loss.
  - **Returns:**
    - bool: Whether early stopping criterion is met.

**Example:**
```python
from sensor_actuator_network import EarlyStopper

# Initialize early stopper
early_stopper = EarlyStopper(patience=10, min_delta=1e-4)

# Simulated training loop
val_losses = [0.5, 0.45, 0.43, 0.41, 0.42, 0.425, 0.422, 0.421, 0.421, 0.42, 0.421, 0.421]

for epoch, val_loss in enumerate(val_losses):
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
    
    # Check if we should stop
    if early_stopper.check(val_loss):
        print(f"Early stopping triggered after epoch {epoch+1}")
        break
```

## Training Functions

### train_model

```python
def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: Optional[DataLoader] = None, 
                epochs: int = 30, 
                lr: float = 1e-3, 
                device: str = "cpu",
                weight_reconstruction: float = 1.0,
                weight_encoding: float = 0.1,
                weight_actuation: float = 1.0,
                checkpoint_dir: Optional[str] = None) -> Dict[str, List[float]]:
    # ...
```

Train the SensorAugmentor model with optional validation.

**Parameters:**
- `model` (nn.Module): Model to train.
- `train_loader` (DataLoader): DataLoader for training data.
- `val_loader` (DataLoader, optional): DataLoader for validation data. Default: None.
- `epochs` (int, optional): Number of training epochs. Default: 30.
- `lr` (float, optional): Learning rate for the optimizer. Default: 1e-3.
- `device` (str, optional): Device to use for training ('cpu' or 'cuda'). Default: 'cpu'.
- `weight_reconstruction` (float, optional): Weight for the reconstruction loss. Default: 1.0.
- `weight_encoding` (float, optional): Weight for the encoding alignment loss. Default: 0.1.
- `weight_actuation` (float, optional): Weight for the actuation loss. Default: 1.0.
- `checkpoint_dir` (str, optional): Directory to save model checkpoints. Default: None.

**Returns:**
- Dict[str, List[float]]: Dictionary containing training and validation losses for each epoch.

**Example:**
```python
import torch
from sensor_actuator_network import SensorAugmentor, SyntheticSensorDataset, train_model
from torch.utils.data import DataLoader, random_split

# Create model and dataset
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
dataset = SyntheticSensorDataset(num_samples=2000)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train model
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=1e-3,
    device=device,
    weight_reconstruction=1.0,
    weight_encoding=0.1,
    weight_actuation=1.0,
    checkpoint_dir="checkpoints"
)

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Training Loss')
if 'val_loss' in history:
    plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training History')
plt.grid(True)
plt.show()
```

### set_seed

```python
def set_seed(seed: int = 42) -> None:
    # ...
```

Sets random seeds for Python, NumPy, and PyTorch for reproducibility.

**Parameters:**
- `seed` (int, optional): Random seed to use. Default: 42.

**Returns:**
- None

**Example:**
```python
from sensor_actuator_network import set_seed

# Set seed for reproducibility
set_seed(123)

# Now all random operations will be deterministic
import torch
import numpy as np

# These will give the same results every time
x1 = torch.randn(5)
x2 = np.random.randn(5)
```

## Advanced Components

### ModelSerializer

```python
class ModelSerializer:
    @staticmethod
    def save_model(model: nn.Module, 
                   path: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        # ...
    
    @staticmethod
    def load_model(path: str, model_class: Type[nn.Module] = SensorAugmentor) -> Tuple[nn.Module, Dict[str, Any]]:
        # ...
```

Utility class for saving and loading models with metadata.

**Methods:**
- `save_model(model, path, metadata=None)`: Save a model to disk with metadata.
  - **Parameters:**
    - `model` (nn.Module): PyTorch model to save.
    - `path` (str): Path to save the model to.
    - `metadata` (Dict[str, Any], optional): Additional metadata to save with the model. Default: None.
  - **Returns:**
    - None
- `load_model(path, model_class=SensorAugmentor)`: Load a model from disk with metadata.
  - **Parameters:**
    - `path` (str): Path to load the model from.
    - `model_class` (Type[nn.Module], optional): Class of the model to load. Default: SensorAugmentor.
  - **Returns:**
    - Tuple[nn.Module, Dict[str, Any]]: Loaded model and metadata.

**Example:**
```python
from sensor_actuator_network import SensorAugmentor, ModelSerializer
import torch

# Create and train a model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
# ... train the model ...

# Save model with metadata
ModelSerializer.save_model(
    model=model,
    path="models/sensor_model_v1.pt",
    metadata={
        "version": "1.0.0",
        "sensor_type": "accelerometer",
        "performance": {"rmse": 0.05, "mae": 0.03},
        "training_config": {"epochs": 20, "lr": 1e-3}
    }
)

# Load model with metadata
loaded_model, metadata = ModelSerializer.load_model("models/sensor_model_v1.pt")

print(f"Model version: {metadata['version']}")
print(f"Sensor type: {metadata['sensor_type']}")
print(f"Training configuration: {metadata['training_config']}")

# Use the loaded model
loaded_model.eval()
with torch.no_grad():
    x = torch.randn(1, 32)
    output = loaded_model(x)
```

### DataNormalizer

```python
class DataNormalizer:
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None):
        # ...
    
    def fit(self, data: torch.Tensor) -> "DataNormalizer":
        # ...
    
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        # ...
    
    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        # ...
```

Utility class for normalizing and denormalizing data.

**Parameters:**
- `mean` (torch.Tensor, optional): Mean value for normalization. If None, will be calculated from data. Default: None.
- `std` (torch.Tensor, optional): Standard deviation for normalization. If None, will be calculated from data. Default: None.

**Attributes:**
- `mean` (torch.Tensor): Mean value for normalization.
- `std` (torch.Tensor): Standard deviation for normalization.

**Methods:**
- `fit(data)`: Calculate mean and standard deviation from data.
  - **Parameters:**
    - `data` (torch.Tensor): Data to calculate statistics from.
  - **Returns:**
    - DataNormalizer: Self for method chaining.
- `normalize(data)`: Normalize data.
  - **Parameters:**
    - `data` (torch.Tensor): Data to normalize.
  - **Returns:**
    - torch.Tensor: Normalized data.
- `denormalize(data)`: Denormalize data.
  - **Parameters:**
    - `data` (torch.Tensor): Data to denormalize.
  - **Returns:**
    - torch.Tensor: Denormalized data.

**Example:**
```python
import torch
from sensor_actuator_network import DataNormalizer

# Create sample data
data = torch.randn(100, 32) * 5 + 10  # Mean ~10, Std ~5

# Create normalizer and fit to data
normalizer = DataNormalizer().fit(data)

# Normalize data
normalized_data = normalizer.normalize(data)
print(f"Normalized mean: {normalized_data.mean()}")  # ~0
print(f"Normalized std: {normalized_data.std()}")    # ~1

# Denormalize data
denormalized_data = normalizer.denormalize(normalized_data)
print(f"Original mean: {data.mean()}")
print(f"Denormalized mean: {denormalized_data.mean()}")  # Should match original

# Create normalizer with predefined statistics
custom_normalizer = DataNormalizer(mean=torch.tensor([10.0]), std=torch.tensor([5.0]))
custom_normalized = custom_normalizer.normalize(torch.tensor([[15.0]]))
print(f"Custom normalized: {custom_normalized}")  # Should be ~1.0
```

### UncertaintyEstimator

```python
class UncertaintyEstimator:
    def __init__(self, model: nn.Module, num_samples: int = 10, dropout_rate: float = 0.1):
        # ...
    
    def estimate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ...
```

Utility class for estimating uncertainty in model predictions using Monte Carlo dropout.

**Parameters:**
- `model` (nn.Module): Model to use for uncertainty estimation.
- `num_samples` (int, optional): Number of samples to use for uncertainty estimation. Default: 10.
- `dropout_rate` (float, optional): Dropout rate to use. Default: 0.1.

**Methods:**
- `estimate(x)`: Estimate uncertainty in model predictions.
  - **Parameters:**
    - `x` (torch.Tensor): Input data of shape (batch_size, sensor_dim).
  - **Returns:**
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mean prediction, standard deviation, and entropy.

**Example:**
```python
import torch
from sensor_actuator_network import SensorAugmentor, UncertaintyEstimator

# Create model with dropout (required for uncertainty estimation)
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)

# Create uncertainty estimator
uncertainty_estimator = UncertaintyEstimator(model, num_samples=20, dropout_rate=0.2)

# Generate sample data
x = torch.randn(5, 32)  # 5 samples, 32-dimensional

# Estimate uncertainty
mean, std, entropy = uncertainty_estimator.estimate(x)

# Print results
for i in range(5):
    print(f"Sample {i+1}:")
    print(f"  Mean prediction: {mean[i]}")
    print(f"  Standard deviation: {std[i]}")
    print(f"  Entropy: {entropy[i]}")
```

## Utility Functions

### visualize_reconstruction

```python
def visualize_reconstruction(original: torch.Tensor, 
                             reconstructed: torch.Tensor, 
                             sensor_names: Optional[List[str]] = None) -> plt.Figure:
    # ...
```

Visualize the comparison between original and reconstructed sensor data.

**Parameters:**
- `original` (torch.Tensor): Original sensor data of shape (batch_size, sensor_dim).
- `reconstructed` (torch.Tensor): Reconstructed sensor data of shape (batch_size, sensor_dim).
- `sensor_names` (List[str], optional): Names of the sensors. If None, numeric indices will be used. Default: None.

**Returns:**
- plt.Figure: Matplotlib figure containing the visualization.

**Example:**
```python
import torch
import matplotlib.pyplot as plt
from sensor_actuator_network import SensorAugmentor, visualize_reconstruction

# Create model
model = SensorAugmentor(sensor_dim=8, hidden_dim=32, output_dim=1)

# Generate sample data
x_hq = torch.randn(1, 8)  # High-quality data
x_lq = x_hq + 0.3 * torch.randn(1, 8)  # Low-quality data with noise

# Generate reconstruction
model.eval()
with torch.no_grad():
    reconstructed, _, _, _ = model(x_lq)

# Visualize reconstruction
sensor_names = ["Temp", "Pressure", "Humidity", "CO2", "PM2.5", "PM10", "VOC", "O3"]
fig = visualize_reconstruction(x_hq, reconstructed, sensor_names)
plt.tight_layout()
plt.show()
```

### evaluate_model

```python
def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  device: str = "cpu") -> Dict[str, float]:
    # ...
```

Evaluate a trained model on a test dataset.

**Parameters:**
- `model` (nn.Module): Trained model to evaluate.
- `test_loader` (DataLoader): DataLoader for test data.
- `device` (str, optional): Device to use for evaluation ('cpu' or 'cuda'). Default: 'cpu'.

**Returns:**
- Dict[str, float]: Dictionary containing evaluation metrics.

**Example:**
```python
import torch
from torch.utils.data import DataLoader, random_split
from sensor_actuator_network import SensorAugmentor, SyntheticSensorDataset, evaluate_model

# Create model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
# ... train the model ...

# Create test dataset
dataset = SyntheticSensorDataset(num_samples=500)
test_loader = DataLoader(dataset, batch_size=32)

# Evaluate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metrics = evaluate_model(model, test_loader, device)

# Print metrics
print("Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.6f}")
```

### calculate_metrics

```python
def calculate_metrics(y_true: torch.Tensor, 
                      y_pred: torch.Tensor) -> Dict[str, float]:
    # ...
```

Calculate various metrics between true and predicted values.

**Parameters:**
- `y_true` (torch.Tensor): Ground truth values.
- `y_pred` (torch.Tensor): Predicted values.

**Returns:**
- Dict[str, float]: Dictionary containing various metrics.

**Example:**
```python
import torch
from sensor_actuator_network import calculate_metrics

# Create sample data
y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = torch.tensor([1.1, 2.2, 2.8, 4.1, 4.7])

# Calculate metrics
metrics = calculate_metrics(y_true, y_pred)

# Print metrics
print("Metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.6f}")
```

---

For more detailed examples and tutorials, see the [examples directory](../examples/) and the [tutorials section](../tutorials/index.md) of the documentation. 