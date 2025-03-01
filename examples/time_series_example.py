"""
Example: Using SensorAugmentor with time-series sensor data.

This example shows how to:
1. Process time-series data from multiple sensors
2. Structure the data for input to the SensorAugmentor
3. Train the model on sequential data
4. Evaluate and visualize the results

The data represents a simulated industrial vibration sensor system
with various qualities of accelerometers monitoring machine health.
"""

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add parent directory to path to import SensorAugmentor
sys.path.append(str(Path(__file__).parent.parent))

from sensor_actuator_network import SensorAugmentor, set_seed


class VibrationSensorDataset(torch.utils.data.Dataset):
    """
    Dataset simulating vibration sensors with different quality levels.
    
    The data represents a machine health monitoring scenario where:
    - Multiple accelerometers measure vibration data over time
    - High-quality sensors (expensive) provide accurate readings
    - Low-quality sensors (affordable) provide noisy readings
    - An actuator command represents required dampening adjustment
    """
    def __init__(self, num_samples=500, seq_length=128, num_sensors=3, frequencies=None):
        super().__init__()
        
        # Default frequencies to simulate if none provided (in Hz)
        if frequencies is None:
            # Primary frequency, harmonic, and a resonance frequency
            frequencies = [10, 20, 35]
        
        # Sampling rate (Hz)
        sampling_rate = 100
        
        # Time values
        time = np.linspace(0, seq_length / sampling_rate, seq_length)
        
        # Initialize data arrays
        # Shape: [num_samples, seq_length, num_sensors]
        hq_data = np.zeros((num_samples, seq_length, num_sensors))
        
        # Generate samples
        for i in range(num_samples):
            # Random amplitudes for each frequency component
            amplitudes = np.random.uniform(0.5, 1.5, len(frequencies))
            # Random phase shifts
            phases = np.random.uniform(0, 2*np.pi, len(frequencies))
            
            # Generate the base signal - sum of sinusoids with different frequencies
            base_signal = np.zeros(seq_length)
            for f, a, p in zip(frequencies, amplitudes, phases):
                base_signal += a * np.sin(2 * np.pi * f * time + p)
            
            # Add unique characteristics for each sensor
            for j in range(num_sensors):
                # Slightly different signal for each sensor (simulating sensor placement)
                sensor_signal = base_signal + 0.2 * np.random.randn(seq_length)
                # Add some unique frequency component for this sensor location
                sensor_specific_freq = np.random.uniform(5, 50)
                sensor_signal += 0.3 * np.sin(2 * np.pi * sensor_specific_freq * time)
                
                # Store in high-quality array
                hq_data[i, :, j] = sensor_signal
        
        # Generate low-quality data by adding significant noise
        # The noise level varies by sensor to simulate different quality sensors
        noise_levels = [0.1, 0.3, 0.5]  # Different noise levels for each sensor
        lq_data = np.copy(hq_data)
        
        for j in range(num_sensors):
            # Add noise to each sensor
            noise = noise_levels[j] * np.random.randn(num_samples, seq_length)
            lq_data[:, :, j] += noise
        
        # Calculate actuator command - this would be a dampening adjustment
        # based on the peak frequency and amplitude of vibration
        y_cmd = np.zeros((num_samples, 1))
        for i in range(num_samples):
            # Simple heuristic: use the max amplitude of the high quality signal
            # across all sensors as the basis for the dampening command
            max_amplitude = np.max(np.abs(hq_data[i]))
            # Scale to a reasonable command range [0, 1]
            y_cmd[i, 0] = np.clip(max_amplitude / 3.0, 0, 1)
        
        # Flatten the time and sensor dimensions for the SensorAugmentor
        # Shape: [num_samples, seq_length * num_sensors]
        self.x_hq = torch.tensor(hq_data.reshape(num_samples, -1), dtype=torch.float32)
        self.x_lq = torch.tensor(lq_data.reshape(num_samples, -1), dtype=torch.float32)
        self.y_cmd = torch.tensor(y_cmd, dtype=torch.float32)
        
        # Store original shapes for reshaping during visualization
        self.seq_length = seq_length
        self.num_sensors = num_sensors
        
        # Normalize data
        self.mean_hq = self.x_hq.mean(dim=0, keepdim=True)
        self.std_hq = self.x_hq.std(dim=0, keepdim=True) + 1e-6
        self.mean_lq = self.x_lq.mean(dim=0, keepdim=True)
        self.std_lq = self.x_lq.std(dim=0, keepdim=True) + 1e-6
        
        self.x_hq_norm = (self.x_hq - self.mean_hq) / self.std_hq
        self.x_lq_norm = (self.x_lq - self.mean_lq) / self.std_lq
    
    def __len__(self):
        return len(self.x_lq)
    
    def __getitem__(self, idx):
        return self.x_lq_norm[idx], self.x_hq_norm[idx], self.y_cmd[idx]
    
    def reshape_to_time_series(self, flat_data):
        """Reshape flat tensor back to [batch, time, sensors] format."""
        batch_size = flat_data.shape[0]
        return flat_data.view(batch_size, self.seq_length, self.num_sensors)


def train_and_evaluate():
    """Train and evaluate the SensorAugmentor on vibration sensor data."""
    set_seed(42)
    
    # Parameters
    seq_length = 128
    num_sensors = 3
    num_samples = 800
    batch_size = 32
    epochs = 25
    
    # Total input dimension is sequence length * number of sensors
    sensor_dim = seq_length * num_sensors
    hidden_dim = 256  # Larger for sequence data
    output_dim = 1  # Dampening command
    
    # Create dataset
    dataset = VibrationSensorDataset(
        num_samples=num_samples,
        seq_length=seq_length,
        num_sensors=num_sensors
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    g = torch.Generator()
    g.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=g
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Create model
    model = SensorAugmentor(
        sensor_dim=sensor_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_resblocks=3  # More residual blocks for complex time-series data
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss and optimizer
    criterion_reconstruction = nn.MSELoss()
    criterion_actuation = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for x_lq, x_hq, y_cmd in train_loader:
            x_lq = x_lq.to(device)
            x_hq = x_hq.to(device)
            y_cmd = y_cmd.to(device)
            
            # Forward pass
            recon_hq, act_cmd, encoded_lq, encoded_hq = model(x_lq, x_hq)
            
            # Losses
            loss_recon = criterion_reconstruction(recon_hq, x_hq)
            loss_act = criterion_actuation(act_cmd, y_cmd)
            loss_latent = criterion_reconstruction(encoded_lq, encoded_hq)
            
            # Combined loss
            loss = loss_recon + loss_act + 0.1 * loss_latent
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * x_lq.size(0)
        
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for x_lq, x_hq, y_cmd in val_loader:
                x_lq = x_lq.to(device)
                x_hq = x_hq.to(device)
                y_cmd = y_cmd.to(device)
                
                recon_hq, act_cmd, encoded_lq, encoded_hq = model(x_lq, x_hq)
                
                loss_recon = criterion_reconstruction(recon_hq, x_hq)
                loss_act = criterion_actuation(act_cmd, y_cmd)
                
                val_loss = loss_recon + loss_act
                epoch_val_loss += val_loss.item() * x_lq.size(0)
        
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
    
    print("Training complete!")
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/time_series_loss.png')
    
    # Evaluate on a random sample
    model.eval()
    
    # Get a random sample from validation set
    sample_idx = np.random.randint(0, len(val_dataset))
    x_lq, x_hq, y_cmd = val_dataset[sample_idx]
    
    # Add batch dimension and move to device
    x_lq = x_lq.unsqueeze(0).to(device)
    x_hq = x_hq.unsqueeze(0).to(device)
    y_cmd = y_cmd.to(device)
    
    # Get model prediction
    with torch.no_grad():
        recon_hq, act_cmd, _, _ = model(x_lq)
    
    # Convert to numpy for plotting
    x_lq_np = x_lq.cpu().numpy()
    x_hq_np = x_hq.cpu().numpy()
    recon_hq_np = recon_hq.cpu().numpy()
    
    # Denormalize
    x_lq_denorm = x_lq_np * dataset.std_lq.numpy() + dataset.mean_lq.numpy()
    x_hq_denorm = x_hq_np * dataset.std_hq.numpy() + dataset.mean_hq.numpy()
    recon_hq_denorm = recon_hq_np * dataset.std_hq.numpy() + dataset.mean_hq.numpy()
    
    # Reshape to [time, sensors]
    x_lq_reshaped = x_lq_denorm.reshape(seq_length, num_sensors)
    x_hq_reshaped = x_hq_denorm.reshape(seq_length, num_sensors)
    recon_hq_reshaped = recon_hq_denorm.reshape(seq_length, num_sensors)
    
    # Plot the time series for each sensor
    fig, axes = plt.subplots(num_sensors, 1, figsize=(12, 9), sharex=True)
    time = np.arange(seq_length)
    
    for i in range(num_sensors):
        axes[i].plot(time, x_lq_reshaped[:, i], 'b-', alpha=0.7, label='LQ Sensor')
        axes[i].plot(time, x_hq_reshaped[:, i], 'g-', label='HQ Sensor')
        axes[i].plot(time, recon_hq_reshaped[:, i], 'r--', label='Reconstructed')
        axes[i].set_title(f'Sensor {i+1}')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True)
        axes[i].legend()
    
    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.savefig('outputs/vibration_reconstruction.png')
    
    print(f"Actuator Command (Dampening): {act_cmd.item():.4f}")
    print("Visualizations saved to 'outputs/' directory")
    

if __name__ == "__main__":
    train_and_evaluate() 