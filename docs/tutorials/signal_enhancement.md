# Tutorial: Signal Enhancement with SensorAugmentor

This tutorial guides you through the process of enhancing low-quality sensor signals using the SensorAugmentor framework. You'll learn how to prepare your data, train a model, and deploy it for real-time signal enhancement.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Theory of Signal Enhancement](#theory-of-signal-enhancement)
- [Setting Up Your Environment](#setting-up-your-environment)
- [Preparing Your Data](#preparing-your-data)
- [Training a Signal Enhancement Model](#training-a-signal-enhancement-model)
- [Evaluating Model Performance](#evaluating-model-performance)
- [Deploying the Model](#deploying-the-model)
- [Advanced Techniques](#advanced-techniques)
- [Troubleshooting](#troubleshooting)
- [Conclusion](#conclusion)

## Introduction

Sensor data often suffers from noise, drift, low resolution, or other quality issues. The SensorAugmentor framework allows you to use deep learning to transform low-quality sensor data into high-quality, precise signals that better represent the physical phenomenon being measured.

This tutorial will walk you through a complete workflow using a vibration sensor example, where we'll enhance noisy, low-resolution vibration data to recover detailed vibration patterns.

## Prerequisites

Before beginning this tutorial, ensure you have:

- Python 3.8 or later
- PyTorch 1.9 or later
- NumPy, Matplotlib, and SciPy
- SensorAugmentor framework installed (see [Installation Guide](../index.md#installation))
- Basic understanding of neural networks and PyTorch

## Theory of Signal Enhancement

### How SensorAugmentor Works

At its core, SensorAugmentor uses a neural network architecture with:

1. An **encoder** that maps low-quality signals to a latent space representation
2. **Residual blocks** that process this representation to extract meaningful features
3. A **reconstructor** that maps from the latent space back to a high-quality signal

The framework is trained using paired examples of low-quality and high-quality signals, learning to map from the former to the latter.

### Learning Objectives

1. To compress the essential information in the signal into a latent representation
2. To remove noise and artifacts present in the low-quality signal
3. To recover fine details that may be lost in the low-quality signal
4. To optionally predict actuator commands based on the enhanced signals

### Signal Enhancement vs. Denoising

While traditional denoising focuses solely on removing noise, signal enhancement goes further by:

- Recovering lost details and resolution
- Correcting systematic errors and drift
- Improving temporal or spatial resolution
- Restoring missing data

## Setting Up Your Environment

Let's start by setting up a dedicated environment for this tutorial:

```bash
# Create a virtual environment
python -m venv sensor_env
source sensor_env/bin/activate  # On Windows: sensor_env\Scripts\activate

# Install dependencies
pip install torch torchvision numpy matplotlib scipy

# Install SensorAugmentor
pip install sensor-augmentor
# or from source
# git clone https://github.com/yourusername/SensorAugmentor.git
# cd SensorAugmentor
# pip install -e .
```

## Preparing Your Data

### Data Requirements

For training, you need:
- Low-quality signals (input)
- Corresponding high-quality signals (target)
- Optional: actuator commands associated with the signals

### Synthetic Data Example

Let's start with synthetic data to understand the workflow:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sensor_actuator_network import SyntheticSensorDataset, SensorAugmentor

# Create synthetic vibration data
def generate_vibration_data(num_samples=1000, seq_length=100, sampling_freq=100):
    # Time vector
    t = np.linspace(0, 1, seq_length)
    
    # Storage for signals
    high_quality = np.zeros((num_samples, seq_length))
    
    # Generate samples with different frequencies and phases
    for i in range(num_samples):
        # Main frequency components
        f1 = np.random.uniform(5, 20)  # Base frequency
        f2 = np.random.uniform(25, 40)  # Higher frequency component
        
        # Amplitudes
        a1 = np.random.uniform(0.7, 1.0)
        a2 = np.random.uniform(0.3, 0.6)
        
        # Phase shifts
        p1 = np.random.uniform(0, 2*np.pi)
        p2 = np.random.uniform(0, 2*np.pi)
        
        # Generate clean signal (sum of sinusoids)
        signal = a1 * np.sin(2*np.pi*f1*t + p1) + a2 * np.sin(2*np.pi*f2*t + p2)
        
        # Add minor noise to high-quality signal
        high_quality[i] = signal + np.random.normal(0, 0.02, seq_length)
    
    # Create low-quality version with issues
    low_quality = np.zeros_like(high_quality)
    for i in range(num_samples):
        # Add significant noise
        noise = np.random.normal(0, 0.2, seq_length)
        
        # Add frequency-dependent attenuation (loss of high frequencies)
        signal_fft = np.fft.rfft(high_quality[i])
        freq = np.fft.rfftfreq(seq_length, 1/sampling_freq)
        filter_response = 1 / (1 + 0.1 * freq)  # Low-pass filter
        filtered_signal = np.fft.irfft(signal_fft * filter_response)
        
        # Add sensor drift (baseline wander)
        drift = 0.2 * np.sin(2*np.pi*0.1*t)
        
        # Combine effects
        low_quality[i] = filtered_signal + noise + drift
    
    # Generate simple actuator commands (for demonstration)
    # In this case, we'll generate a command that's related to the 
    # amplitude and frequency characteristics of the signal
    actuator_commands = np.zeros((num_samples, 2))
    for i in range(num_samples):
        # Extract features from the high-quality signal
        fft_vals = np.abs(np.fft.rfft(high_quality[i]))
        peak_freq_idx = np.argmax(fft_vals[1:]) + 1
        peak_amplitude = np.max(high_quality[i]) - np.min(high_quality[i])
        
        # Map these to actuator commands (e.g., damping and stiffness)
        actuator_commands[i, 0] = 0.5 + 0.5 * peak_amplitude  # Damping
        actuator_commands[i, 1] = 0.2 + 0.8 * (peak_freq_idx / (len(fft_vals) - 1))  # Stiffness
    
    return torch.tensor(low_quality, dtype=torch.float32), torch.tensor(high_quality, dtype=torch.float32), torch.tensor(actuator_commands, dtype=torch.float32)

# Generate data
x_lq, x_hq, y_cmd = generate_vibration_data(num_samples=5000, seq_length=128)

# Visualize a sample
sample_idx = 42  # Choose any sample
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x_hq[sample_idx].numpy(), label='High-Quality Signal')
plt.title('High-Quality Vibration Signal')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x_lq[sample_idx].numpy(), label='Low-Quality Signal')
plt.title('Low-Quality Vibration Signal (Noisy, Attenuated, with Drift)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Actuator commands for this sample: {y_cmd[sample_idx].numpy()}")
```

### Real-World Data Preparation

For real-world data, you typically need to:

1. **Synchronize your signals**: Ensure low-quality and high-quality signals are properly time-aligned
2. **Normalize your data**: Scale signals to a similar range, typically [-1, 1] or [0, 1]
3. **Split your data**: Create training, validation, and test sets

```python
# Example of normalizing and splitting real data
from sklearn.model_selection import train_test_split
from sensor_actuator_network import DataNormalizer

class VibrationDataset(torch.utils.data.Dataset):
    def __init__(self, x_lq, x_hq, y_cmd=None):
        self.x_lq = x_lq
        self.x_hq = x_hq
        self.y_cmd = y_cmd
    
    def __len__(self):
        return len(self.x_lq)
    
    def __getitem__(self, idx):
        if self.y_cmd is not None:
            return self.x_lq[idx], self.x_hq[idx], self.y_cmd[idx]
        else:
            return self.x_lq[idx], self.x_hq[idx]

# Split data
x_lq_train, x_lq_test, x_hq_train, x_hq_test, y_cmd_train, y_cmd_test = train_test_split(
    x_lq, x_hq, y_cmd, test_size=0.2, random_state=42
)

# Further split test into validation and test
x_lq_val, x_lq_test, x_hq_val, x_hq_test, y_cmd_val, y_cmd_test = train_test_split(
    x_lq_test, x_hq_test, y_cmd_test, test_size=0.5, random_state=42
)

# Normalize data using the training set statistics
normalizer = DataNormalizer()
normalizer.fit(x_lq_train)  # Calculate mean and std from training data

# Apply normalization
x_lq_train_norm = normalizer.normalize(x_lq_train)
x_lq_val_norm = normalizer.normalize(x_lq_val)
x_lq_test_norm = normalizer.normalize(x_lq_test)

# No need to normalize target data in this case as it's already in a good range
# But if needed:
# hq_normalizer = DataNormalizer()
# hq_normalizer.fit(x_hq_train)
# x_hq_train_norm = hq_normalizer.normalize(x_hq_train)
# x_hq_val_norm = hq_normalizer.normalize(x_hq_val)
# x_hq_test_norm = hq_normalizer.normalize(x_hq_test)

# Create datasets
train_dataset = VibrationDataset(x_lq_train_norm, x_hq_train, y_cmd_train)
val_dataset = VibrationDataset(x_lq_val_norm, x_hq_val, y_cmd_val)
test_dataset = VibrationDataset(x_lq_test_norm, x_hq_test, y_cmd_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

## Training a Signal Enhancement Model

Now that we have our data prepared, let's train a SensorAugmentor model:

```python
import torch.nn.functional as F
from sensor_actuator_network import SensorAugmentor, EarlyStopper, set_seed

# Set seed for reproducibility
set_seed(42)

# Define model parameters
sensor_dim = x_lq.shape[1]  # Number of timesteps in your signal
hidden_dim = 256            # Size of latent space
output_dim = y_cmd.shape[1] # Dimension of actuator commands
num_resblocks = 3           # Number of residual blocks

# Create model
model = SensorAugmentor(
    sensor_dim=sensor_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_resblocks=num_resblocks
)

# Define optimizer and learning rate
lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Define loss function
def calculate_loss(reconstructed_hq, act_command, encoded_lq, encoded_hq, x_hq, y_cmd):
    # Reconstruction loss (mean squared error)
    loss_recon = F.mse_loss(reconstructed_hq, x_hq)
    
    # Encoding similarity loss
    loss_encoding = F.mse_loss(encoded_lq, encoded_hq)
    
    # Actuator command prediction loss
    loss_act = F.mse_loss(act_command, y_cmd)
    
    # Combined loss with weights
    total_loss = loss_recon + 0.1 * loss_encoding + 0.5 * loss_act
    
    return total_loss, loss_recon, loss_encoding, loss_act

# Training loop
num_epochs = 100
patience = 10  # For early stopping
early_stopper = EarlyStopper(patience=patience)

# Track losses
train_losses = []
val_losses = []
recon_losses = []
encoding_losses = []
act_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for x_lq, x_hq, y_cmd in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed_hq, act_command, encoded_lq, encoded_hq = model(x_lq, x_hq)
        
        # Calculate loss
        loss, loss_recon, loss_encoding, loss_act = calculate_loss(
            reconstructed_hq, act_command, encoded_lq, encoded_hq, x_hq, y_cmd
        )
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_recon_loss = 0.0
    val_encoding_loss = 0.0
    val_act_loss = 0.0
    
    with torch.no_grad():
        for x_lq, x_hq, y_cmd in val_loader:
            reconstructed_hq, act_command, encoded_lq, encoded_hq = model(x_lq, x_hq)
            loss, loss_recon, loss_encoding, loss_act = calculate_loss(
                reconstructed_hq, act_command, encoded_lq, encoded_hq, x_hq, y_cmd
            )
            val_loss += loss.item()
            val_recon_loss += loss_recon.item()
            val_encoding_loss += loss_encoding.item()
            val_act_loss += loss_act.item()
    
    val_loss /= len(val_loader)
    val_recon_loss /= len(val_loader)
    val_encoding_loss /= len(val_loader)
    val_act_loss /= len(val_loader)
    
    val_losses.append(val_loss)
    recon_losses.append(val_recon_loss)
    encoding_losses.append(val_encoding_loss)
    act_losses.append(val_act_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"  Recon Loss: {val_recon_loss:.4f}, Encoding Loss: {val_encoding_loss:.4f}, Act Loss: {val_act_loss:.4f}")
    
    # Early stopping
    if early_stopper.early_stop(val_loss):
        print(f"Early stopping at epoch {epoch+1}")
        break

# Save model with normalization parameters
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'normalizer_mean': normalizer.mean,
    'normalizer_std': normalizer.std,
    'config': {
        'sensor_dim': sensor_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_resblocks': num_resblocks
    }
}, 'vibration_enhancer_model.pt')

# Plot training progress
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(recon_losses, label='Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Signal Reconstruction Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(encoding_losses, label='Encoding Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Latent Space Encoding Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(act_losses, label='Actuator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Actuator Command Prediction Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Evaluating Model Performance

After training, let's evaluate the model's performance on the test set:

```python
from sensor_actuator_network import calculate_metrics, visualize_reconstruction

# Set model to evaluation mode
model.eval()

# Containers for predictions and ground truth
all_lq = []
all_hq = []
all_pred_hq = []
all_cmd = []
all_pred_cmd = []

# Evaluate on test set
with torch.no_grad():
    for x_lq, x_hq, y_cmd in test_loader:
        # Forward pass
        reconstructed_hq, act_command, _, _ = model(x_lq)
        
        # Store results
        all_lq.append(x_lq)
        all_hq.append(x_hq)
        all_pred_hq.append(reconstructed_hq)
        all_cmd.append(y_cmd)
        all_pred_cmd.append(act_command)

# Concatenate results
all_lq = torch.cat(all_lq, dim=0)
all_hq = torch.cat(all_hq, dim=0)
all_pred_hq = torch.cat(all_pred_hq, dim=0)
all_cmd = torch.cat(all_cmd, dim=0)
all_pred_cmd = torch.cat(all_pred_cmd, dim=0)

# Calculate metrics for signal reconstruction
recon_metrics = calculate_metrics(all_hq, all_pred_hq)
print("Signal Reconstruction Metrics:")
print(f"  MSE: {recon_metrics['mse']:.6f}")
print(f"  MAE: {recon_metrics['mae']:.6f}")
print(f"  RMSE: {recon_metrics['rmse']:.6f}")
print(f"  PSNR: {recon_metrics['psnr']:.2f} dB")
print(f"  SSIM: {recon_metrics['ssim']:.4f}")

# Calculate metrics for actuator command prediction
cmd_metrics = calculate_metrics(all_cmd, all_pred_cmd)
print("\nActuator Command Prediction Metrics:")
print(f"  MSE: {cmd_metrics['mse']:.6f}")
print(f"  MAE: {cmd_metrics['mae']:.6f}")
print(f"  RMSE: {cmd_metrics['rmse']:.6f}")

# Visualize some examples
num_examples = 3
for i in range(num_examples):
    idx = np.random.randint(0, len(all_lq))
    
    plt.figure(figsize=(12, 8))
    
    # Plot signals
    plt.subplot(2, 1, 1)
    plt.plot(all_lq[idx].numpy(), label='Low-Quality Signal', alpha=0.7)
    plt.plot(all_hq[idx].numpy(), label='True High-Quality Signal')
    plt.plot(all_pred_hq[idx].numpy(), label='Reconstructed Signal', linestyle='--')
    plt.title(f'Signal Reconstruction Example {i+1}')
    plt.legend()
    plt.grid(True)
    
    # Plot actuator commands
    plt.subplot(2, 1, 2)
    
    # Bar chart for actuator commands (assuming small dimension)
    x = np.arange(len(all_cmd[idx]))
    width = 0.35
    plt.bar(x - width/2, all_cmd[idx].numpy(), width, label='True Commands')
    plt.bar(x + width/2, all_pred_cmd[idx].numpy(), width, label='Predicted Commands')
    plt.title(f'Actuator Command Prediction Example {i+1}')
    plt.xticks(x, [f'Cmd {j+1}' for j in range(len(all_cmd[idx]))])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Visualize reconstruction in time-frequency domain
for i in range(3):
    idx = np.random.randint(0, len(all_lq))
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Time domain signals
    plt.subplot(3, 2, 1)
    plt.plot(all_lq[idx].numpy(), label='Low-Quality')
    plt.title('Low-Quality Signal (Time Domain)')
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    plt.plot(all_hq[idx].numpy(), label='High-Quality')
    plt.title('True High-Quality Signal (Time Domain)')
    plt.grid(True)
    
    plt.subplot(3, 2, 5)
    plt.plot(all_pred_hq[idx].numpy(), label='Reconstructed')
    plt.title('Reconstructed Signal (Time Domain)')
    plt.grid(True)
    
    # Compute spectrograms
    from scipy import signal as sg
    
    def compute_spectrogram(x):
        f, t, Sxx = sg.spectrogram(x, fs=100, nperseg=32, noverlap=16)
        return f, t, Sxx
    
    # Low-quality spectrogram
    f, t, Sxx_lq = compute_spectrogram(all_lq[idx].numpy())
    plt.subplot(3, 2, 2)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx_lq), shading='gouraud')
    plt.title('Low-Quality Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='PSD [dB]')
    
    # High-quality spectrogram
    f, t, Sxx_hq = compute_spectrogram(all_hq[idx].numpy())
    plt.subplot(3, 2, 4)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx_hq), shading='gouraud')
    plt.title('True High-Quality Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.colorbar(label='PSD [dB]')
    
    # Reconstructed spectrogram
    f, t, Sxx_pred = compute_spectrogram(all_pred_hq[idx].numpy())
    plt.subplot(3, 2, 6)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx_pred), shading='gouraud')
    plt.title('Reconstructed Spectrogram')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='PSD [dB]')
    
    plt.tight_layout()
    plt.show()
```

## Deploying the Model

Once you're satisfied with your model's performance, you can deploy it for real-time inference:

### Simple Inference Function

```python
def enhance_signal(model, normalizer, low_quality_signal):
    """
    Enhance a low-quality signal using the trained model.
    
    Args:
        model: Trained SensorAugmentor model
        normalizer: DataNormalizer used during training
        low_quality_signal: Tensor or numpy array of shape (signal_length,)
    
    Returns:
        Tuple of (enhanced_signal, actuator_commands)
    """
    # Ensure input is a PyTorch tensor
    if isinstance(low_quality_signal, np.ndarray):
        low_quality_signal = torch.tensor(low_quality_signal, dtype=torch.float32)
    
    # Add batch dimension if needed
    if len(low_quality_signal.shape) == 1:
        low_quality_signal = low_quality_signal.unsqueeze(0)
    
    # Normalize input
    normalized_signal = normalizer.normalize(low_quality_signal)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        reconstructed_hq, actuator_commands, _, _ = model(normalized_signal)
    
    # Return enhanced signal and actuator commands
    return reconstructed_hq.squeeze(0), actuator_commands.squeeze(0)
```

### Basic Real-Time Processing Example

```python
def real_time_signal_processor(model, normalizer, input_stream, output_stream, buffer_size=128, overlap=64):
    """
    Process a stream of signal data in real-time.
    
    Args:
        model: Trained SensorAugmentor model
        normalizer: DataNormalizer used during training
        input_stream: Function that yields chunks of signal data
        output_stream: Function that consumes enhanced chunks of signal data
        buffer_size: Size of processing buffer
        overlap: Overlap between consecutive buffers for smooth transitions
    """
    # Initialize buffers
    prev_buffer = None
    prev_enhanced = None
    
    # Process stream
    for chunk in input_stream():
        # If this is the first chunk, initialize with zeros
        if prev_buffer is None:
            prev_buffer = torch.zeros(overlap)
        
        # Combine with previous buffer to handle overlap
        current_buffer = torch.cat([prev_buffer, chunk])
        
        # Process buffer
        enhanced_signal, actuator_commands = enhance_signal(model, normalizer, current_buffer)
        
        # If this is the first chunk, output the entire enhanced buffer
        if prev_enhanced is None:
            output_chunk = enhanced_signal
        else:
            # Use a linear crossfade in the overlap region
            crossfade = torch.linspace(0, 1, overlap)
            overlap_region = (1 - crossfade) * prev_enhanced[-overlap:] + crossfade * enhanced_signal[:overlap]
            output_chunk = torch.cat([prev_enhanced[:-overlap], overlap_region, enhanced_signal[overlap:]])
        
        # Send to output stream
        output_stream(output_chunk, actuator_commands)
        
        # Save for next iteration
        prev_buffer = chunk[-overlap:]
        prev_enhanced = enhanced_signal
```

### REST API for Signal Enhancement

For deploying as a service, you can use FastAPI:

```python
# Install with: pip install fastapi uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
from sensor_actuator_network import SensorAugmentor, DataNormalizer

# Create FastAPI app
app = FastAPI(title="Signal Enhancement API", 
             description="API for enhancing sensor signals using SensorAugmentor")

# Load model and normalizer
checkpoint = torch.load('vibration_enhancer_model.pt', map_location='cpu')
model = SensorAugmentor(
    sensor_dim=checkpoint['config']['sensor_dim'],
    hidden_dim=checkpoint['config']['hidden_dim'],
    output_dim=checkpoint['config']['output_dim'],
    num_resblocks=checkpoint['config']['num_resblocks']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

normalizer = DataNormalizer(
    mean=checkpoint['normalizer_mean'],
    std=checkpoint['normalizer_std']
)

# Define request/response models
class SensorData(BaseModel):
    signal: List[float]

class EnhancementResponse(BaseModel):
    enhanced_signal: List[float]
    actuator_commands: List[float]

@app.post("/enhance", response_model=EnhancementResponse)
async def enhance_signal_endpoint(data: SensorData):
    try:
        # Convert to tensor
        signal = torch.tensor(data.signal, dtype=torch.float32)
        
        # Check if signal length matches expected dimension
        if len(signal) != model.sensor_dim:
            raise HTTPException(
                status_code=400, 
                detail=f"Signal length must be {model.sensor_dim}, but got {len(signal)}"
            )
        
        # Add batch dimension
        signal = signal.unsqueeze(0)
        
        # Normalize
        normalized_signal = normalizer.normalize(signal)
        
        # Process
        with torch.no_grad():
            reconstructed, actuator_commands, _, _ = model(normalized_signal)
        
        # Return results
        return EnhancementResponse(
            enhanced_signal=reconstructed[0].tolist(),
            actuator_commands=actuator_commands[0].tolist()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn signal_enhancement_api:app --reload
```

## Advanced Techniques

### Fine-tuning for Specific Sensors

You can fine-tune a pre-trained model for your specific sensor:

```python
# Load pre-trained model
checkpoint = torch.load('pretrained_model.pt')
model = SensorAugmentor(
    sensor_dim=checkpoint['config']['sensor_dim'],
    hidden_dim=checkpoint['config']['hidden_dim'],
    output_dim=checkpoint['config']['output_dim'],
    num_resblocks=checkpoint['config']['num_resblocks']
)
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze encoder parameters (if you want to keep the learned representations)
for param in model.encoder.parameters():
    param.requires_grad = False

# Only train the reconstructor and actuator head
optimizer = torch.optim.Adam([
    {'params': model.hq_reconstructor.parameters()},
    {'params': model.actuator_head.parameters()}
], lr=1e-4)

# Fine-tune on your specific dataset
# ... (similar to the training loop above)
```

### Transfer Learning Between Sensor Types

You can transfer knowledge between different sensor types:

```python
# Load a model trained on accelerometer data
accel_model = SensorAugmentor(...)
accel_model.load_state_dict(torch.load('accelerometer_model.pt')['model_state_dict'])

# Create a new model for gyroscope data
gyro_model = SensorAugmentor(
    sensor_dim=gyro_dim,  # Possibly different dimension
    hidden_dim=accel_model.hidden_dim,  # Same latent space dimension
    output_dim=gyro_output_dim
)

# Copy encoder weights (assuming dimensions match)
gyro_model.encoder.load_state_dict(accel_model.encoder.state_dict())
gyro_model.post_encoding_resblock.load_state_dict(accel_model.post_encoding_resblock.state_dict())

# Train with the encoder frozen initially
for param in gyro_model.encoder.parameters():
    param.requires_grad = False

# Train for a few epochs with encoder frozen
# ...

# Then unfreeze and continue training
for param in gyro_model.encoder.parameters():
    param.requires_grad = True

# Continue training with lower learning rate
# ...
```

### Uncertainty Estimation

You can enhance your model to provide uncertainty estimates:

```python
from sensor_actuator_network import SensorAugmentorWithDropout, UncertaintyEstimator

# Create model with dropout
model = SensorAugmentorWithDropout(
    sensor_dim=sensor_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_resblocks=num_resblocks,
    dropout_rate=0.1
)

# Train the model as before

# Create uncertainty estimator
uncertainty_estimator = UncertaintyEstimator(model, num_samples=30)

# Get prediction with uncertainty
mean_prediction, std_deviation = uncertainty_estimator.predict_with_uncertainty(input_signal)

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(input_signal.numpy(), label='Low-Quality Input', alpha=0.7)
plt.plot(mean_prediction.numpy(), label='Enhanced Signal')

# Add uncertainty band (±2 std deviations = 95% confidence interval)
plt.fill_between(
    range(len(mean_prediction)),
    (mean_prediction - 2 * std_deviation).numpy(),
    (mean_prediction + 2 * std_deviation).numpy(),
    alpha=0.3,
    label='95% Confidence Interval'
)
plt.title('Signal Enhancement with Uncertainty Estimation')
plt.legend()
plt.grid(True)
plt.show()
```

## Troubleshooting

### Common Issues and Solutions

1. **Model not converging**
   - Try a lower learning rate (e.g., 1e-4 instead of 1e-3)
   - Increase model capacity (larger `hidden_dim` or more `num_resblocks`)
   - Check data normalization (ensure inputs are properly scaled)
   - Inspect your data for outliers or inconsistencies

2. **High training loss but poor reconstruction**
   - Your loss weights may need adjustment (increase the weight of `loss_recon`)
   - There might be insufficient correlation between low-quality and high-quality signals
   - Try a different model architecture with more capacity

3. **Memory issues**
   - Reduce batch size
   - Decrease model size (smaller `hidden_dim`)
   - Use gradient accumulation for effectively larger batches

4. **Poor generalization to new data**
   - Add more diverse training examples
   - Implement data augmentation techniques
   - Add regularization (L2 weight decay or dropout)

### Debugging Tools

```python
# Debug model gradients
def debug_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:")
            print(f"  Mean: {param.data.mean().item():.6f}")
            print(f"  Std: {param.data.std().item():.6f}")
            if param.grad is not None:
                print(f"  Grad Mean: {param.grad.mean().item():.6f}")
                print(f"  Grad Std: {param.grad.std().item():.6f}")
                print(f"  Grad Min: {param.grad.min().item():.6f}")
                print(f"  Grad Max: {param.grad.max().item():.6f}")
            else:
                print("  No gradient")
            print()

# Call during training
debug_gradients(model)
```

## Conclusion

In this tutorial, you've learned how to:

1. Prepare sensor data for training a signal enhancement model
2. Train a SensorAugmentor model to enhance low-quality signals
3. Evaluate the model's performance using various metrics
4. Deploy the model for real-time signal enhancement
5. Apply advanced techniques like transfer learning and uncertainty estimation

SensorAugmentor provides a flexible framework for enhancing various types of sensor data. By adapting the techniques in this tutorial to your specific sensors, you can significantly improve signal quality and derive more accurate insights from your sensor data.

For more information and advanced use cases, refer to:
- [API Reference](../api/index.md)
- [Architecture Documentation](../architecture/index.md)
- [Deployment Guide](../deployment/index.md) 