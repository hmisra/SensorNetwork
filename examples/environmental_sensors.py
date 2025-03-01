"""
Environmental Sensors Example

This example shows how to use SensorAugmentor to enhance environmental sensor data
and generate actuator commands for environmental control systems.

The example demonstrates:
1. Creating a custom dataset for environmental sensors
2. Training a SensorAugmentor model
3. Evaluating the model performance
4. Using the model for inference
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from datetime import datetime, timedelta

from sensor_actuator_network import SensorAugmentor

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EnvironmentalSensorDataset(torch.utils.data.Dataset):
    """
    Custom dataset simulating environmental sensors with:
    - Temperature sensors (°C)
    - Humidity sensors (%)
    - Air quality (particulate matter, PM2.5)
    - Light levels (lux)
    
    The dataset simulates:
    - Daily and seasonal patterns
    - Sensor noise and drift
    - Low-quality and high-quality sensor variants
    - Optimal actuator commands for environmental control
    """
    
    def __init__(self, num_samples=1000, time_period_days=30, 
                noise_factor_lq=0.2, drift_factor_lq=0.1):
        """
        Initialize the environmental sensor dataset.
        
        Args:
            num_samples: Number of data points to generate
            time_period_days: Time period in days to simulate
            noise_factor_lq: Noise factor for low-quality sensors
            drift_factor_lq: Drift factor for low-quality sensors
        """
        self.num_samples = num_samples
        self.time_period_days = time_period_days
        self.noise_factor_lq = noise_factor_lq
        self.drift_factor_lq = drift_factor_lq
        
        # Sensor parameters
        self.sensor_dim = 4  # Temperature, humidity, air quality, light
        self.actuator_dim = 3  # HVAC power, humidifier power, air purifier power
        
        # Generate timestamps for the data points
        start_date = datetime(2023, 1, 1)
        self.timestamps = [start_date + timedelta(
            days=self.time_period_days * i / num_samples) 
            for i in range(num_samples)]
        
        # Generate clean sensor data based on natural patterns
        self._generate_clean_data()
        
        # Add noise and drift to create low-quality sensor readings
        self._create_lq_sensor_data()
        
        # Generate optimal actuator commands
        self._generate_actuator_commands()
    
    def _generate_clean_data(self):
        """Generate clean environmental sensor data with natural patterns."""
        # Initialize arrays
        self.hq_sensor_data = np.zeros((self.num_samples, self.sensor_dim))
        
        for i, timestamp in enumerate(self.timestamps):
            # Hour of day (0-23)
            hour = timestamp.hour + timestamp.minute / 60
            # Day of year (0-364)
            day_of_year = timestamp.timetuple().tm_yday - 1
            
            # Temperature: Daily cycle + seasonal cycle (°C)
            daily_temp_cycle = -3 * np.cos(2 * np.pi * hour / 24)  # ±3°C daily variation
            seasonal_temp_cycle = 10 * np.cos(2 * np.pi * day_of_year / 365)  # ±10°C seasonal variation
            base_temp = 20  # Base temperature 20°C
            temperature = base_temp + daily_temp_cycle + seasonal_temp_cycle
            
            # Humidity: Inverse relation with temperature, plus daily cycle (%)
            humidity_seasonal = -0.5 * seasonal_temp_cycle + 60
            humidity_daily = 5 * np.sin(2 * np.pi * hour / 24)
            humidity = humidity_seasonal + humidity_daily
            humidity = np.clip(humidity, 30, 85)  # Clip to reasonable values
            
            # Air quality (PM2.5): Worse during morning/evening rush hours
            rush_hour_effect = 25 * (np.exp(-((hour - 8) ** 2) / 4) + np.exp(-((hour - 18) ** 2) / 4))
            weekend_effect = 0.7 if timestamp.weekday() >= 5 else 1.0  # Better air quality on weekends
            air_quality = 10 + rush_hour_effect * weekend_effect
            
            # Light level (lux): Natural daylight cycle
            daylight = np.maximum(0, 1000 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0)
            artificial_light = 300 if 7 <= hour <= 22 else 0  # Indoor lighting when people are awake
            light_level = daylight + artificial_light
            
            # Combine all sensor readings
            self.hq_sensor_data[i, 0] = temperature
            self.hq_sensor_data[i, 1] = humidity
            self.hq_sensor_data[i, 2] = air_quality
            self.hq_sensor_data[i, 3] = light_level
    
    def _create_lq_sensor_data(self):
        """Create low-quality sensor readings by adding noise and drift."""
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_factor_lq, self.hq_sensor_data.shape)
        
        # Add sensor drift (increasing with time)
        drift_pattern = np.linspace(0, 1, self.num_samples).reshape(-1, 1)
        drift = self.drift_factor_lq * drift_pattern * np.random.randn(1, self.sensor_dim)
        
        # Combine clean data with noise and drift
        self.lq_sensor_data = self.hq_sensor_data + noise + drift
        
        # Add specific sensor limitations
        # Low-quality temperature sensor has lower resolution
        self.lq_sensor_data[:, 0] = np.round(self.lq_sensor_data[:, 0] * 2) / 2  # 0.5°C resolution
        
        # Low-quality humidity sensor has capped range
        self.lq_sensor_data[:, 1] = np.clip(self.lq_sensor_data[:, 1], 35, 80)  # Limited range
        
        # Low-quality air quality sensor has poor low-end sensitivity
        self.lq_sensor_data[:, 2] = np.maximum(15, self.lq_sensor_data[:, 2])  # Can't measure below 15
        
        # Low-quality light sensor has non-linear response
        self.lq_sensor_data[:, 3] = np.power(self.lq_sensor_data[:, 3] / 1000, 1.2) * 1000  # Non-linear
    
    def _generate_actuator_commands(self):
        """Generate optimal actuator commands based on high-quality sensor data."""
        self.actuator_commands = np.zeros((self.num_samples, self.actuator_dim))
        
        for i in range(self.num_samples):
            temperature = self.hq_sensor_data[i, 0]
            humidity = self.hq_sensor_data[i, 1]
            air_quality = self.hq_sensor_data[i, 2]
            
            # HVAC power (0-1): Optimal is to maintain temperature around 21-23°C
            target_temp = 22
            temp_diff = temperature - target_temp
            hvac_power = min(1.0, max(0.0, 0.3 * abs(temp_diff))) * (1 if temp_diff > 0 else -1)
            
            # Humidifier power (0-1): Optimal humidity is 40-60%
            if humidity < 40:
                humidifier_power = 0.5 * (40 - humidity) / 10  # Increase if too dry
            elif humidity > 60:
                humidifier_power = -0.5 * (humidity - 60) / 10  # Decrease if too humid
            else:
                humidifier_power = 0
            humidifier_power = min(1.0, max(-1.0, humidifier_power))
            
            # Air purifier power (0-1): Higher when air quality is poor
            air_purifier_power = min(1.0, max(0.0, (air_quality - 15) / 35))
            
            self.actuator_commands[i, 0] = hvac_power
            self.actuator_commands[i, 1] = humidifier_power
            self.actuator_commands[i, 2] = air_purifier_power
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        lq_tensor = torch.tensor(self.lq_sensor_data[idx], dtype=torch.float32)
        hq_tensor = torch.tensor(self.hq_sensor_data[idx], dtype=torch.float32)
        actuator_tensor = torch.tensor(self.actuator_commands[idx], dtype=torch.float32)
        
        return {
            'lq_sensor': lq_tensor,
            'hq_sensor': hq_tensor,
            'actuator_command': actuator_tensor,
            'timestamp': self.timestamps[idx]
        }


def train_environmental_model():
    """Train and evaluate the SensorAugmentor on environmental sensor data."""
    # Initialize dataset
    print("Generating environmental sensor dataset...")
    dataset = EnvironmentalSensorDataset(
        num_samples=5000,
        time_period_days=60,
        noise_factor_lq=0.2,
        drift_factor_lq=0.1
    )
    
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Create model
    print("Initializing SensorAugmentor model...")
    model = SensorAugmentor(
        sensor_dim=4,  # Temperature, humidity, air quality, light
        hidden_dim=64,
        output_dim=3,  # HVAC, humidifier, air purifier
        num_residual_blocks=4
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Starting training...")
    num_epochs = 50
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            lq_sensor = batch['lq_sensor']
            hq_sensor = batch['hq_sensor']
            actuator_command = batch['actuator_command']
            
            # Forward pass
            optimizer.zero_grad()
            hq_reconstruction, predicted_actuator, latent_lq, _ = model(lq_sensor)
            
            # Optional teacher forcing (if you have high-quality sensors during training)
            _, _, latent_hq, _ = model(hq_sensor)
            
            # Calculate losses
            reconstruction_loss = torch.nn.functional.mse_loss(hq_reconstruction, hq_sensor)
            actuator_loss = torch.nn.functional.mse_loss(predicted_actuator, actuator_command)
            alignment_loss = torch.nn.functional.mse_loss(latent_lq, latent_hq.detach())
            
            # Combined loss
            loss = reconstruction_loss + actuator_loss + 0.1 * alignment_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                lq_sensor = batch['lq_sensor']
                hq_sensor = batch['hq_sensor']
                actuator_command = batch['actuator_command']
                
                hq_reconstruction, predicted_actuator, _, _ = model(lq_sensor)
                
                reconstruction_loss = torch.nn.functional.mse_loss(hq_reconstruction, hq_sensor)
                actuator_loss = torch.nn.functional.mse_loss(predicted_actuator, actuator_command)
                
                val_loss += (reconstruction_loss + actuator_loss).item()
        
        val_losses.append(val_loss / len(val_loader))
        
        # Save best model
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/environmental_sensor_model.pth')
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {train_losses[-1]:.4f}, "
                  f"Val Loss: {val_losses[-1]:.4f}")
    
    print("Training completed!")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/environmental_training_curve.png')
    plt.close()
    
    return model, dataset


def evaluate_model(model, dataset):
    """Evaluate the trained model and visualize results."""
    # Create test loader with sequential samples
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Prepare arrays for storing results
    timestamps = []
    lq_values = {
        'Temperature': [], 'Humidity': [], 
        'Air Quality': [], 'Light': []
    }
    hq_values = {
        'Temperature': [], 'Humidity': [], 
        'Air Quality': [], 'Light': []
    }
    reconstructed_values = {
        'Temperature': [], 'Humidity': [], 
        'Air Quality': [], 'Light': []
    }
    actual_commands = {
        'HVAC': [], 'Humidifier': [], 'Air Purifier': []
    }
    predicted_commands = {
        'HVAC': [], 'Humidifier': [], 'Air Purifier': []
    }
    
    # Model evaluation
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 500:  # Only use a subset for visualization
                break
                
            lq_sensor = batch['lq_sensor']
            hq_sensor = batch['hq_sensor']
            actuator_command = batch['actuator_command']
            timestamp = batch['timestamp'][0]
            
            hq_reconstruction, predicted_actuator, _, _ = model(lq_sensor)
            
            # Store results
            timestamps.append(timestamp)
            
            # Sensor values
            lq_values['Temperature'].append(lq_sensor[0, 0].item())
            lq_values['Humidity'].append(lq_sensor[0, 1].item())
            lq_values['Air Quality'].append(lq_sensor[0, 2].item())
            lq_values['Light'].append(lq_sensor[0, 3].item())
            
            hq_values['Temperature'].append(hq_sensor[0, 0].item())
            hq_values['Humidity'].append(hq_sensor[0, 1].item())
            hq_values['Air Quality'].append(hq_sensor[0, 2].item())
            hq_values['Light'].append(hq_sensor[0, 3].item())
            
            reconstructed_values['Temperature'].append(hq_reconstruction[0, 0].item())
            reconstructed_values['Humidity'].append(hq_reconstruction[0, 1].item())
            reconstructed_values['Air Quality'].append(hq_reconstruction[0, 2].item())
            reconstructed_values['Light'].append(hq_reconstruction[0, 3].item())
            
            # Actuator commands
            actual_commands['HVAC'].append(actuator_command[0, 0].item())
            actual_commands['Humidifier'].append(actuator_command[0, 1].item())
            actual_commands['Air Purifier'].append(actuator_command[0, 2].item())
            
            predicted_commands['HVAC'].append(predicted_actuator[0, 0].item())
            predicted_commands['Humidifier'].append(predicted_actuator[0, 1].item())
            predicted_commands['Air Purifier'].append(predicted_actuator[0, 2].item())
    
    # Create visualizations
    os.makedirs('results', exist_ok=True)
    
    # Convert timestamps to numerical format for plotting
    plot_dates = [ts.timestamp() for ts in timestamps]
    date_labels = [ts.strftime('%m-%d %H:%M') for ts in timestamps[::50]]
    date_ticks = [plot_dates[i] for i in range(0, len(plot_dates), 50)]
    
    # Plot sensor readings
    for sensor_name in ['Temperature', 'Humidity', 'Air Quality', 'Light']:
        plt.figure(figsize=(12, 6))
        plt.plot(plot_dates, lq_values[sensor_name], 'r-', alpha=0.6, label='Low-Quality')
        plt.plot(plot_dates, hq_values[sensor_name], 'g-', label='High-Quality (Ground Truth)')
        plt.plot(plot_dates, reconstructed_values[sensor_name], 'b--', label='Reconstructed')
        
        plt.title(f'{sensor_name} Sensor Readings Over Time')
        plt.xlabel('Time')
        plt.ylabel('Sensor Value')
        plt.xticks(date_ticks, date_labels, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/environmental_{sensor_name.lower()}_comparison.png')
        plt.close()
    
    # Plot actuator commands
    for actuator_name in ['HVAC', 'Humidifier', 'Air Purifier']:
        plt.figure(figsize=(12, 6))
        plt.plot(plot_dates, actual_commands[actuator_name], 'g-', label='Optimal Command')
        plt.plot(plot_dates, predicted_commands[actuator_name], 'b--', label='Predicted Command')
        
        plt.title(f'{actuator_name} Control Commands Over Time')
        plt.xlabel('Time')
        plt.ylabel('Actuator Power (-1 to 1)')
        plt.xticks(date_ticks, date_labels, rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/environmental_{actuator_name.lower().replace(" ", "_")}_commands.png')
        plt.close()
    
    # Calculate and print metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    
    # Calculate RMSE for each sensor
    for sensor_name in ['Temperature', 'Humidity', 'Air Quality', 'Light']:
        hq = np.array(hq_values[sensor_name])
        recon = np.array(reconstructed_values[sensor_name])
        rmse = np.sqrt(np.mean((hq - recon) ** 2))
        print(f"{sensor_name} Reconstruction RMSE: {rmse:.4f}")
    
    # Calculate RMSE for actuator commands
    for actuator_name in ['HVAC', 'Humidifier', 'Air Purifier']:
        actual = np.array(actual_commands[actuator_name])
        predicted = np.array(predicted_commands[actuator_name])
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        print(f"{actuator_name} Command RMSE: {rmse:.4f}")


if __name__ == "__main__":
    print("Environmental Sensor Enhancement Example")
    print("=" * 40)
    
    # Train the model
    trained_model, env_dataset = train_environmental_model()
    
    # Evaluate and visualize results
    evaluate_model(trained_model, env_dataset)
    
    print("\nExample completed successfully. Results saved to 'results/' directory.") 