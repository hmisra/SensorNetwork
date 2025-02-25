# Sensor Actuator Network

## Overview
This repository demonstrates a neural network system (SensorAugmentor) that:
1. Learns to reconstruct high-quality (HQ) signals from lower-quality (LQ) sensor inputs.
2. Produces a refined latent representation for downstream tasks, such as actuator control.
3. Uses residual connections and a teacher-student style approach for robust training.

## Key Features
1. **Residual Blocks**: Improves signal flow and stabilizes deep architectures.
2. **Teacher-Student Alignment**: Aligns latent encodings of LQ and HQ signals, enabling the network to "upgrade" low-quality signals.
3. **Data Normalization**: Normalizes the dataset for more stable training.
4. **Early Stopping & LR Scheduling**: Automatically handles learning rate adjustments and prevents overfitting.
5. **Single/Multi-GPU Compatibility**: Includes a note on using DataParallel for multi-GPU setups.
6. **Reproducibility**: Global seeding for Python, NumPy, and Torch. Also includes a seeded train-validation split.

## Structure
- **set_seed**(seed=42): Ensures consistent random number generation.
- **ResidualBlock**(dim): A basic residual block with two linear layers.
- **SensorAugmentor**: The primary neural network.
  - Encoder: Transforms sensor data into a latent representation.
  - HQ Reconstructor: Recreates HQ signals from the latent representation.
  - Actuator Head: Regresses or classifies an actuator command.
- **SyntheticSensorDataset**: A synthetic dataset that pairs LQ and HQ sensor data, plus a ground-truth actuator command.
- **EarlyStopper**: Monitors validation loss for early stopping.
- **train_model**: Manages the training loop, including LR scheduling and early stopping.
- **main**: Example usage demonstration.

## Usage Instructions
1. Install Dependencies:
   - PyTorch
   - NumPy

2. Run the Script:
   ```bash
   python sensor_actuator_network.py
   ```
3. Training:
   - The code automatically trains on GPU if available; otherwise it falls back to CPU.
   - Adjust hyperparameters (epochs, batch_size, etc.) in the `main()` function.

4. Inference:
   - The script demonstrates how to use the trained network to "upgrade" new LQ sensor readings and produce an actuator command.

## Extending
- **Multi-GPU Training**: Wrap the model with `nn.DataParallel(model)` or `DistributedDataParallel`.
- **Alternate Loss Functions**: Replace MSE with CrossEntropyLoss for classification tasks.
- **Additional Residual Blocks**: Increase `num_resblocks` for deeper architectures.
- **Real Data**: Integrate real sensor streams or time-series data in place of `SyntheticSensorDataset`.
