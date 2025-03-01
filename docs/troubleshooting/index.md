# Troubleshooting Guide

This guide helps you diagnose and solve common issues you might encounter when working with the SensorAugmentor framework.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data Preparation Problems](#data-preparation-problems)
- [Training Issues](#training-issues)
- [Model Performance Issues](#model-performance-issues)
- [Inference Problems](#inference-problems)
- [Deployment Challenges](#deployment-challenges)
- [Platform-Specific Issues](#platform-specific-issues)
- [Getting Help](#getting-help)

## Installation Issues

### PyTorch Installation Failures

**Problem**: Unable to install PyTorch or getting errors during installation.

**Solution**:

1. Verify you're using the command from the [official PyTorch website](https://pytorch.org/get-started/locally/) for your specific OS/CUDA configuration
2. For CUDA compatibility issues:
   ```bash
   # Check CUDA version
   nvcc --version
   # or
   nvidia-smi
   ```
3. Install PyTorch with a specific CUDA version:
   ```bash
   # For CUDA 11.7
   pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   ```
4. If GPU support isn't necessary, use CPU-only version:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
   ```

### Package Version Conflicts

**Problem**: Dependencies conflict with existing packages in your environment.

**Solution**:

1. Create a dedicated virtual environment:
   ```bash
   python -m venv sensor_env
   source sensor_env/bin/activate  # On Windows: sensor_env\Scripts\activate
   ```
2. Install with specific versions:
   ```bash
   pip install -r requirements.txt
   ```
3. If conflicts persist, try installing packages one by one, starting with PyTorch

### Import Errors

**Problem**: `ImportError: No module named 'sensor_actuator_network'` when trying to use the package.

**Solution**:

1. Verify the module is installed:
   ```bash
   pip list | grep sensor
   ```
2. Check your Python path:
   ```python
   import sys
   print(sys.path)
   ```
3. Ensure you're running Python from the correct environment
4. Install in development mode:
   ```bash
   pip install -e .
   ```

## Data Preparation Problems

### Incorrect Dimension Errors

**Problem**: Getting dimension mismatch errors when feeding data to the model.

**Solution**:

1. Check the expected dimensions:
   ```python
   print(f"Model expects input shape: [batch_size, {model.sensor_dim}]")
   print(f"Your data shape: {data.shape}")
   ```
2. Reshape your data correctly:
   ```python
   # For single samples, add batch dimension
   if len(data.shape) == 1:
       data = data.unsqueeze(0)  # Add batch dimension
   ```
3. Transpose data if needed:
   ```python
   # If your data is [features, samples] instead of [samples, features]
   data = data.T
   ```

### Normalization Issues

**Problem**: Model performs poorly due to data not being normalized correctly.

**Solution**:

1. Verify normalization statistics:
   ```python
   print(f"Mean: {data.mean()}, Std: {data.std()}")
   ```
2. Apply correct normalization:
   ```python
   normalized_data = (data - mean) / std
   ```
3. Use the provided `DataNormalizer` class:
   ```python
   from sensor_actuator_network import DataNormalizer
   
   normalizer = DataNormalizer().fit(train_data)
   normalized_train = normalizer.normalize(train_data)
   normalized_test = normalizer.normalize(test_data)
   ```
4. Make sure to save normalization parameters with the model for inference:
   ```python
   model_info = {
       "model_state_dict": model.state_dict(),
       "normalization": {
           "mean": normalizer.mean,
           "std": normalizer.std
       }
   }
   torch.save(model_info, "model.pt")
   ```

### Data Type Issues

**Problem**: TypeError or unexpected behavior due to incorrect data types.

**Solution**:

1. Ensure data is the correct type for PyTorch:
   ```python
   # Convert numpy arrays to PyTorch tensors
   if isinstance(data, np.ndarray):
       data = torch.from_numpy(data).float()
   
   # Ensure correct data type
   data = data.to(torch.float32)
   ```
2. Check for NaN or Inf values:
   ```python
   if torch.isnan(data).any() or torch.isinf(data).any():
       print("Warning: Data contains NaN or Inf values")
       # Handle by replacing with zeros or mean values
       data = torch.where(torch.isnan(data) | torch.isinf(data), 
                         torch.zeros_like(data), data)
   ```

## Training Issues

### Loss Not Decreasing

**Problem**: Training loss stays flat or doesn't decrease significantly.

**Solution**:

1. Check your learning rate:
   ```python
   # Try a different learning rate
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Try lower value like 1e-4
   ```
2. Verify data is normalized properly (see Normalization Issues above)
3. Inspect gradients for vanishing/exploding issues:
   ```python
   # Add this to your training loop to monitor gradients
   for name, param in model.named_parameters():
       if param.requires_grad:
           print(f"{name}: grad_min={param.grad.min()}, grad_max={param.grad.max()}")
   ```
4. Try a different optimization algorithm:
   ```python
   # Try SGD instead of Adam
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
   ```
5. Check if the model has enough capacity:
   ```python
   # Increase model capacity
   model = SensorAugmentor(sensor_dim=32, hidden_dim=128, num_resblocks=4)
   ```

### Training Divergence

**Problem**: Loss suddenly spikes or becomes NaN during training.

**Solution**:

1. Add gradient clipping:
   ```python
   # Add to your training loop right after loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```
2. Lower your learning rate:
   ```python
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   ```
3. Use learning rate scheduling:
   ```python
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.5, patience=5
   )
   # In your validation loop
   scheduler.step(val_loss)
   ```
4. Check for extreme values in your data

### Out of Memory (OOM) Errors

**Problem**: CUDA out of memory errors during training.

**Solution**:

1. Reduce batch size:
   ```python
   # Try a smaller batch size
   train_loader = DataLoader(dataset, batch_size=16)  # Reduce from default
   ```
2. Use gradient accumulation for effective larger batch sizes:
   ```python
   accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
   optimizer.zero_grad()
   for i, (x_lq, x_hq, y_cmd) in enumerate(train_loader):
       # Forward pass and loss calculation
       loss = ...
       loss = loss / accumulation_steps  # Normalize loss
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```
3. Move some operations to CPU if necessary:
   ```python
   # Process data preparation on CPU
   preprocessed_data = heavy_preprocessing(data.cpu())
   # Move back to GPU for model
   preprocessed_data = preprocessed_data.to(device)
   ```
4. Use mixed precision training (for Volta GPUs and newer):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   for x_lq, x_hq, y_cmd in train_loader:
       x_lq, x_hq, y_cmd = x_lq.to(device), x_hq.to(device), y_cmd.to(device)
       
       # Enables autocasting for this forward pass
       with autocast():
           reconstructed_hq, act_command, encoded_lq, encoded_hq = model(x_lq, x_hq)
           loss = calculate_loss(...)
       
       optimizer.zero_grad()
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

## Model Performance Issues

### Poor Reconstruction Quality

**Problem**: Model doesn't reconstruct high-quality signals well.

**Solution**:

1. Adjust loss function weights:
   ```python
   # Increase weight for reconstruction loss
   loss = 2.0 * loss_recon + 0.1 * loss_encoding + 0.5 * loss_act
   ```
2. Increase model capacity:
   ```python
   model = SensorAugmentor(
       sensor_dim=32, 
       hidden_dim=128,  # Increase from default 64
       num_resblocks=4  # Increase from default 2
   )
   ```
3. Check if there's enough correlation between LQ and HQ signals:
   ```python
   correlation = np.corrcoef(
       dataset.x_lq.numpy().reshape(-1), 
       dataset.x_hq.numpy().reshape(-1)
   )[0, 1]
   print(f"LQ-HQ correlation: {correlation}")
   # If close to 0, your sensors may be too different
   ```
4. Add more layers to the reconstructor:
   ```python
   class EnhancedSensorAugmentor(SensorAugmentor):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           
           # Replace hq_reconstructor with a deeper network
           self.hq_reconstructor = nn.Sequential(
               nn.Linear(self.hidden_dim, self.hidden_dim),
               nn.ReLU(),
               nn.Linear(self.hidden_dim, self.hidden_dim),
               nn.ReLU(),
               nn.Linear(self.hidden_dim, self.sensor_dim)
           )
   ```

### Poor Actuator Command Prediction

**Problem**: Model doesn't predict actuator commands accurately.

**Solution**:

1. Adjust loss function weights:
   ```python
   # Increase weight for actuator loss
   loss = 1.0 * loss_recon + 0.1 * loss_encoding + 2.0 * loss_act
   ```
2. Ensure actuator commands are properly normalized
3. Check the relation between sensor data and actuator commands:
   ```python
   # Create a simple model to check if sensor data predicts actuator commands
   from sklearn.linear_model import LinearRegression
   
   X = dataset.x_hq.numpy()
   y = dataset.y_cmd.numpy()
   
   reg = LinearRegression().fit(X, y)
   print(f"Simple model score: {reg.score(X, y)}")
   # If score is very low, there might not be enough signal in the data
   ```
4. Enhance the actuator head:
   ```python
   class EnhancedSensorAugmentor(SensorAugmentor):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           
           # Replace actuator_head with a deeper network
           self.actuator_head = nn.Sequential(
               nn.Linear(self.hidden_dim, self.hidden_dim),
               nn.ReLU(),
               nn.Linear(self.hidden_dim, self.output_dim)
           )
   ```

### Overfitting

**Problem**: Model performs well on training data but poorly on validation data.

**Solution**:

1. Add regularization:
   ```python
   # Add weight decay to optimizer
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
   ```
2. Add dropout:
   ```python
   class SensorAugmentorWithDropout(SensorAugmentor):
       def __init__(self, dropout_rate=0.2, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.dropout = nn.Dropout(dropout_rate)
           
       def forward(self, x_lq, x_hq=None):
           # Same as original but with dropout
           encoded_lq = self.encoder(x_lq)
           encoded_lq = self.post_encoding_resblock(encoded_lq)
           encoded_lq = self.dropout(encoded_lq)  # Add dropout
           
           # Rest of forward pass
           # ...
   ```
3. Use early stopping (already implemented in the library)
4. Increase training data or add data augmentation:
   ```python
   def augment_sensor_data(data, noise_level=0.05):
       """Add small random noise to sensor data for augmentation."""
       noise = noise_level * torch.randn_like(data)
       return data + noise
   
   # In training loop
   x_lq_augmented = augment_sensor_data(x_lq)
   x_hq_augmented = augment_sensor_data(x_hq)
   ```

## Inference Problems

### Shape Mismatch During Inference

**Problem**: Getting shape mismatch errors during inference.

**Solution**:

1. Check input shape:
   ```python
   print(f"Expected input shape: [batch_size, {model.sensor_dim}]")
   print(f"Your input shape: {input_data.shape}")
   ```
2. Ensure your input is correctly batched:
   ```python
   # For single sample, add batch dimension
   if len(input_data.shape) == 1:
       input_data = input_data.unsqueeze(0)
   ```
3. For multi-sample prediction, use proper batching:
   ```python
   batch_size = 32
   predictions = []
   
   for i in range(0, len(input_data), batch_size):
       batch = input_data[i:i+batch_size]
       with torch.no_grad():
           batch_predictions = model(batch)
       predictions.append(batch_predictions[0])  # Assuming you want reconstructed_hq
   
   # Concatenate results
   all_predictions = torch.cat(predictions, dim=0)
   ```

### Missing Normalization During Inference

**Problem**: Poor results because input data isn't normalized correctly.

**Solution**:

1. Always normalize inputs using the same statistics as during training:
   ```python
   # Load model with normalization parameters
   checkpoint = torch.load("model.pt")
   model.load_state_dict(checkpoint["model_state_dict"])
   mean = checkpoint["normalization"]["mean"]
   std = checkpoint["normalization"]["std"]
   
   # Normalize input
   normalized_input = (input_data - mean) / std
   
   # Inference
   with torch.no_grad():
       output = model(normalized_input)
   ```
2. Use the provided `DataNormalizer` class for consistency:
   ```python
   # During training
   normalizer = DataNormalizer().fit(train_data)
   torch.save({
       "model_state_dict": model.state_dict(),
       "normalizer_mean": normalizer.mean,
       "normalizer_std": normalizer.std
   }, "model.pt")
   
   # During inference
   checkpoint = torch.load("model.pt")
   model.load_state_dict(checkpoint["model_state_dict"])
   normalizer = DataNormalizer(
       mean=checkpoint["normalizer_mean"],
       std=checkpoint["normalizer_std"]
   )
   
   # Normalize and predict
   normalized_input = normalizer.normalize(input_data)
   output = model(normalized_input)
   ```

### Slow Inference

**Problem**: Model inference is too slow for your application.

**Solution**:

1. Use batch processing (see Shape Mismatch solution #3)
2. Optimize model for inference:
   ```python
   # Set model to evaluation mode
   model.eval()
   
   # Use torch.no_grad() to disable gradient calculation
   with torch.no_grad():
       output = model(input_data)
   ```
3. Try model quantization for faster CPU inference:
   ```python
   # Quantize model
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   
   # Use quantized model
   with torch.no_grad():
       output = quantized_model(input_data)
   ```
4. Export to TorchScript for C++ deployment:
   ```python
   # Export to TorchScript
   scripted_model = torch.jit.script(model)
   scripted_model.save("model_scripted.pt")
   ```
5. Use GPU if available:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   input_data = input_data.to(device)
   ```

## Deployment Challenges

### Model Size Issues

**Problem**: Model is too large for your deployment environment.

**Solution**:

1. Quantize the model to reduce size:
   ```python
   # Dynamic quantization
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   
   # Save quantized model
   torch.save(quantized_model.state_dict(), "model_quantized.pt")
   
   # Check size reduction
   import os
   original_size = os.path.getsize("model.pt") / 1024
   quantized_size = os.path.getsize("model_quantized.pt") / 1024
   print(f"Original size: {original_size:.2f} KB")
   print(f"Quantized size: {quantized_size:.2f} KB")
   print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
   ```
2. Prune unnecessary parameters:
   ```python
   # Simple magnitude-based pruning
   from torch.nn.utils import prune
   
   # Prune 20% of smallest weights in all linear layers
   for name, module in model.named_modules():
       if isinstance(module, torch.nn.Linear):
           prune.l1_unstructured(module, name='weight', amount=0.2)
   
   # Make pruning permanent
   for name, module in model.named_modules():
       if isinstance(module, torch.nn.Linear):
           prune.remove(module, 'weight')
   ```
3. Use a smaller model architecture:
   ```python
   # Reduce model size
   smaller_model = SensorAugmentor(
       sensor_dim=32,
       hidden_dim=32,  # Reduced from default 64
       num_resblocks=1  # Reduced from default 2
   )
   ```

### API Integration Issues

**Problem**: Difficulties integrating the model into a REST API.

**Solution**:

1. Use FastAPI for easy integration:
   ```python
   from fastapi import FastAPI, HTTPException
   from pydantic import BaseModel
   from typing import List
   import torch
   import numpy as np
   from sensor_actuator_network import SensorAugmentor
   
   app = FastAPI()
   
   # Load model (do this outside the request handlers for efficiency)
   checkpoint = torch.load("model.pt", map_location=torch.device('cpu'))
   model = SensorAugmentor(
       sensor_dim=checkpoint["config"]["sensor_dim"],
       hidden_dim=checkpoint["config"]["hidden_dim"],
       output_dim=checkpoint["config"]["output_dim"]
   )
   model.load_state_dict(checkpoint["model_state_dict"])
   model.eval()
   
   # Get normalization parameters
   mean = checkpoint["normalization"]["mean"]
   std = checkpoint["normalization"]["std"]
   
   class SensorData(BaseModel):
       data: List[float]
   
   @app.post("/predict")
   def predict(sensor_data: SensorData):
       try:
           # Convert to tensor
           data = torch.tensor([sensor_data.data], dtype=torch.float32)
           
           # Normalize
           normalized_data = (data - mean) / std
           
           # Inference
           with torch.no_grad():
               reconstructed_hq, actuator_command, _, _ = model(normalized_data)
           
           return {
               "reconstructed_hq": reconstructed_hq[0].tolist(),
               "actuator_command": actuator_command[0].tolist()
           }
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   ```
2. For high-throughput applications, use a queue:
   ```python
   # requirements.txt
   # fastapi
   # uvicorn
   # redis
   # rq
   
   # worker.py
   import torch
   from sensor_actuator_network import SensorAugmentor
   from redis import Redis
   from rq import Worker, Queue, Connection
   
   # Load model
   model = SensorAugmentor(...)
   model.load_state_dict(torch.load("model.pt")["model_state_dict"])
   model.eval()
   
   def process_prediction(data):
       tensor_data = torch.tensor(data, dtype=torch.float32)
       with torch.no_grad():
           output = model(tensor_data)
       return output[0].tolist()  # Return reconstructed_hq
   
   # Start worker
   redis_conn = Redis()
   with Connection(redis_conn):
       worker = Worker(Queue('sensor_predictions'))
       worker.work()
   
   # api.py
   from fastapi import FastAPI
   from redis import Redis
   from rq import Queue
   
   app = FastAPI()
   q = Queue('sensor_predictions', connection=Redis())
   
   @app.post("/predict_async")
   def predict_async(data: dict):
       job = q.enqueue('worker.process_prediction', data["sensor_data"])
       return {"job_id": job.id}
   
   @app.get("/result/{job_id}")
   def get_result(job_id: str):
       job = q.fetch_job(job_id)
       if job.is_finished:
           return {"result": job.result}
       elif job.is_failed:
           return {"status": "failed", "error": job.exc_info}
       else:
           return {"status": "pending"}
   ```

### Containerization Issues

**Problem**: Issues with Docker containerization.

**Solution**:

1. Use the official PyTorch Docker image as a base:
   ```dockerfile
   FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   CMD ["python", "api_server.py"]
   ```
2. For deployment size issues, use multi-stage builds:
   ```dockerfile
   # Build stage
   FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime AS builder
   
   WORKDIR /build
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir --target=/install -r requirements.txt
   
   # Runtime stage
   FROM python:3.8-slim
   
   WORKDIR /app
   
   COPY --from=builder /install /usr/local/lib/python3.8/site-packages
   COPY models/ /app/models/
   COPY sensor_actuator_network.py /app/
   COPY api_server.py /app/
   
   CMD ["python", "api_server.py"]
   ```
3. Ensure model files are correctly included in the image:
   ```dockerfile
   # Make sure models directory exists in the image
   RUN mkdir -p /app/models
   
   # Copy model files
   COPY models/sensor_model.pt /app/models/
   ```

## Platform-Specific Issues

### CUDA Issues on Windows

**Problem**: CUDA errors when running on Windows.

**Solution**:

1. Ensure matching CUDA toolkit and PyTorch versions:
   ```bash
   # Check PyTorch CUDA version
   python -c "import torch; print(torch.version.cuda)"
   
   # Check system CUDA version
   nvcc --version
   ```
2. Add CUDA DLLs to PATH:
   ```bash
   # Add to Windows PATH
   set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin
   ```
3. Try CPU-only version for debugging:
   ```python
   # Force CPU usage
   model = model.to('cpu')
   data = data.to('cpu')
   ```

### macOS Deployment

**Problem**: Issues deploying on macOS.

**Solution**:

1. For macOS, use CPU-only version as CUDA is not supported
2. For Apple Silicon (M1/M2), use PyTorch with MPS support:
   ```python
   if torch.backends.mps.is_available():
       device = torch.device("mps")
   else:
       device = torch.device("cpu")
   
   model = model.to(device)
   data = data.to(device)
   ```
3. Handle Metal Performance Shaders (MPS) specific issues:
   ```python
   # Some operations might not be supported on MPS
   # Fall back to CPU for these
   try:
       output = model(data.to(device))
   except RuntimeError as e:
       if "not implemented for" in str(e) and device.type == "mps":
           print("Operation not supported on MPS, falling back to CPU")
           model = model.to("cpu")
           output = model(data.to("cpu"))
           model = model.to(device)  # Move back to MPS
       else:
           raise
   ```

### Linux Server Deployment

**Problem**: Issues deploying on Linux servers.

**Solution**:

1. Ensure correct library versions:
   ```bash
   # Check CUDA compatibility
   ldconfig -p | grep cuda
   
   # Install required libraries for PyTorch
   sudo apt-get install -y libopenblas-dev libomp-dev
   ```
2. Set environment variables:
   ```bash
   # Set number of OpenMP threads
   export OMP_NUM_THREADS=4
   
   # Disable NUMA balancing for better performance
   echo 0 | sudo tee /proc/sys/kernel/numa_balancing
   ```
3. Handle headless servers (no display):
   ```python
   # Set matplotlib to use a non-GUI backend
   import matplotlib
   matplotlib.use('Agg')
   ```

## Getting Help

If you're still experiencing issues after trying the solutions in this guide:

1. Check the [GitHub Issues](https://github.com/yourusername/SensorAugmentor/issues) to see if someone has reported a similar problem
2. Search the [Discussions forum](https://github.com/yourusername/SensorAugmentor/discussions) for related topics
3. Create a detailed issue report including:
   - SensorAugmentor version
   - Python/PyTorch versions
   - OS and hardware information
   - Complete error message and stack trace
   - Minimal reproducible example

For urgent issues or commercial support, contact [support@sensoraugmentor.ai](mailto:support@sensoraugmentor.ai).

---

This troubleshooting guide covers common issues you might encounter. For more specific problems, please refer to the [documentation](../index.md) or reach out to the community. 