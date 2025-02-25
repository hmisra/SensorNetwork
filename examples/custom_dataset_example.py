"""
Example: Using SensorAugmentor with a custom dataset.

This example shows how to:
1. Create a custom dataset for your sensor data
2. Train the SensorAugmentor with your data 
3. Save and load the model
4. Use the model for inference

The custom dataset in this example simulates time-series data from
a temperature and humidity sensor array with different qualities.
"""

import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import SensorAugmentor
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sensor_actuator_network import SensorAugmentor, set_seed


class EnvironmentalSensorDataset(torch.utils.data.Dataset):
    """
    Custom dataset simulating environmental sensors with:
    - Temperature readings (in Celsius)
    - Humidity readings (in percentage)
    - Pressure readings (in hPa)
    - Light levels (in lux)
    
    We create pairs of high-quality and low-quality sensor readings,
    and an actuator command representing HVAC control.
    """
    def __init__(self, num_samples=1000, time_steps=24, noise_levels=(0.2, 0.5, 0.3, 0.4)):
        super().__init__()
        
        # Set dimensions: [batch, time, sensor_type]
        # 4 sensor types: temperature, humidity, pressure, light
        self.sensor_dim = 4 * time_steps  # Flattened sensor readings across time
        
        # Generate daily patterns for environmental factors
        time = np.linspace(0, 2*np.pi, time_steps)
        
        # Base data for high-quality sensors
        samples_hq = []
        for _ in range(num_samples):
            # Temperature: 20-25°C with daily oscillation
            temp_base = 22 + 3 * np.sin(time + 0.5 * np.random.randn())
            
            # Humidity: 40-60% with inverse relation to temperature
            humid_base = 50 - 10 * np.sin(time + 0.5 * np.random.randn())
            
            # Pressure: 1000-1020 hPa with slow variation
            pressure_base = 1010 + 5 * np.sin(0.5 * time + 0.3 * np.random.randn())
            
            # Light: 0-1000 lux with day/night cycle
            light_base = 500 + 500 * np.sin(time - 1.0 + 0.1 * np.random.randn())
            light_base = np.maximum(light_base, 0)  # No negative light
            
            # Stack and flatten
            sample = np.stack([temp_base, humid_base, pressure_base, light_base], axis=1)
            samples_hq.append(sample.flatten())
            
        # Convert to tensor
        self.x_hq = torch.tensor(np.array(samples_hq), dtype=torch.float32)
        
        # Generate low-quality sensor data by adding noise
        temp_noise, humid_noise, pressure_noise, light_noise = noise_levels
        noise_factors = np.array([temp_noise, humid_noise, pressure_noise, light_noise])
        
        # Apply noise to each sensor type
        noise = torch.zeros_like(self.x_hq)
        for i, noise_level in enumerate(noise_factors):
            # Add noise to each sensor type
            indices = slice(i, self.sensor_dim, 4)  # Every 4th element
            noise[:, indices] = noise_level * torch.randn_like(self.x_hq[:, indices])
        
        self.x_lq = self.x_hq + noise
        
        # Normalize data
        self.mean_hq = self.x_hq.mean(dim=0, keepdim=True)
        self.std_hq = self.x_hq.std(dim=0, keepdim=True) + 1e-6
        self.mean_lq = self.x_lq.mean(dim=0, keepdim=True)
        self.std_lq = self.x_lq.std(dim=0, keepdim=True) + 1e-6
        
        self.x_hq_norm = (self.x_hq - self.mean_hq) / self.std_hq
        self.x_lq_norm = (self.x_lq - self.mean_lq) / self.std_lq
        
        # Create actuator command: control signal for HVAC
        # Based on temperature and humidity
        # Extract temperature and humidity from time_steps in the middle
        mid_idx = time_steps // 2
        temp_idx = mid_idx * 4
        humid_idx = temp_idx + 1
        
        # A simple HVAC control policy
        hvac_control = (self.x_hq_norm[:, temp_idx] - 0.5) + (self.x_hq_norm[:, humid_idx] - 0.5)
        self.y_cmd = hvac_control.unsqueeze(1)  # Shape: [batch, 1]
    
    def __len__(self):
        return self.x_lq.size(0)
    
    def __getitem__(self, idx):
        return self.x_lq_norm[idx], self.x_hq_norm[idx], self.y_cmd[idx]


def train_and_evaluate():
    """Train and evaluate the SensorAugmentor on environmental sensor data."""
    set_seed(42)
    
    # Parameters
    num_samples = 1500
    time_steps = 24
    batch_size = 32
    epochs = 30
    
    # Dataset
    dataset = EnvironmentalSensorDataset(
        num_samples=num_samples,
        time_steps=time_steps,
        noise_levels=(0.2, 0.5, 0.3, 0.4)
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
    
    # Model
    sensor_dim = dataset.sensor_dim
    hidden_dim = 128
    output_dim = 1
    
    model = SensorAugmentor(
        sensor_dim=sensor_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_resblocks=3
    )
    
    # Loss and optimizer
    criterion_reconstruction = nn.MSELoss()
    criterion_actuation = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training loop
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for x_lq, x_hq, y_cmd in train_loader:
            x_lq = x_lq.to(device)
            x_hq = x_hq.to(device)
            y_cmd = y_cmd.to(device)
            
            # Forward
            recon_hq, act_cmd, encoded_lq, encoded_hq = model(x_lq, x_hq)
            
            # Loss
            loss_recon = criterion_reconstruction(recon_hq, x_hq)
            loss_enc = criterion_reconstruction(encoded_lq, encoded_hq)
            loss_act = criterion_actuation(act_cmd, y_cmd)
            
            loss = loss_recon + 0.1 * loss_enc + loss_act
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x_lq.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_lq, x_hq, y_cmd in val_loader:
                x_lq = x_lq.to(device)
                x_hq = x_hq.to(device)
                y_cmd = y_cmd.to(device)
                
                recon_hq, act_cmd, encoded_lq, encoded_hq = model(x_lq, x_hq)
                
                loss_recon = criterion_reconstruction(recon_hq, x_hq)
                loss_enc = criterion_reconstruction(encoded_lq, encoded_hq)
                loss_act = criterion_actuation(act_cmd, y_cmd)
                
                batch_loss = loss_recon + 0.1 * loss_enc + loss_act
                val_loss += batch_loss.item() * x_lq.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/environmental_sensor_model.pth')
    
    # Visualization of validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/validation_loss.png')
    
    # Visualization of example reconstruction
    model.eval()
    
    # Get a random sample from validation set
    sample_idx = np.random.randint(0, len(val_dataset))
    x_lq, x_hq, y_cmd = val_dataset[sample_idx]
    
    x_lq = x_lq.unsqueeze(0).to(device)
    x_hq = x_hq.unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        recon_hq, act_cmd, _, _ = model(x_lq)
    
    # Convert tensors to numpy
    x_lq_np = x_lq.cpu().squeeze().numpy()
    x_hq_np = x_hq.cpu().squeeze().numpy()
    recon_hq_np = recon_hq.cpu().squeeze().numpy()
    
    # Unnormalize
    x_lq_orig = x_lq_np * dataset.std_lq.numpy() + dataset.mean_lq.numpy()
    x_hq_orig = x_hq_np * dataset.std_hq.numpy() + dataset.mean_hq.numpy()
    recon_hq_orig = recon_hq_np * dataset.std_hq.numpy() + dataset.mean_hq.numpy()
    
    # Plot temperature for a single day (first 24 values with stride 4)
    temperature_indices = np.arange(0, dataset.sensor_dim, 4)
    
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(time_steps)
    
    plt.subplot(2, 2, 1)
    plt.plot(time_axis, x_lq_orig[temperature_indices], 'b-', label='LQ Sensor')
    plt.plot(time_axis, x_hq_orig[temperature_indices], 'g-', label='HQ Sensor')
    plt.plot(time_axis, recon_hq_orig[temperature_indices], 'r--', label='Reconstructed')
    plt.title('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    
    # Plot humidity
    humidity_indices = np.arange(1, dataset.sensor_dim, 4)
    
    plt.subplot(2, 2, 2)
    plt.plot(time_axis, x_lq_orig[humidity_indices], 'b-', label='LQ Sensor')
    plt.plot(time_axis, x_hq_orig[humidity_indices], 'g-', label='HQ Sensor')
    plt.plot(time_axis, recon_hq_orig[humidity_indices], 'r--', label='Reconstructed')
    plt.title('Humidity (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot pressure
    pressure_indices = np.arange(2, dataset.sensor_dim, 4)
    
    plt.subplot(2, 2, 3)
    plt.plot(time_axis, x_lq_orig[pressure_indices], 'b-', label='LQ Sensor')
    plt.plot(time_axis, x_hq_orig[pressure_indices], 'g-', label='HQ Sensor')
    plt.plot(time_axis, recon_hq_orig[pressure_indices], 'r--', label='Reconstructed')
    plt.title('Pressure (hPa)')
    plt.legend()
    plt.grid(True)
    
    # Plot light
    light_indices = np.arange(3, dataset.sensor_dim, 4)
    
    plt.subplot(2, 2, 4)
    plt.plot(time_axis, x_lq_orig[light_indices], 'b-', label='LQ Sensor')
    plt.plot(time_axis, x_hq_orig[light_indices], 'g-', label='HQ Sensor')
    plt.plot(time_axis, recon_hq_orig[light_indices], 'r--', label='Reconstructed')
    plt.title('Light (lux)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/sensor_reconstruction.png')
    
    print(f"Actuator Command (HVAC Control): {act_cmd.item():.4f}")
    print("Visualizations saved to 'outputs/' directory")


if __name__ == "__main__":
    train_and_evaluate() 