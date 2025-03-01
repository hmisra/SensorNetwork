# Deployment Guide

This guide provides detailed instructions for deploying SensorAugmentor models in production environments. We cover multiple deployment scenarios including cloud services, edge devices, and containerized solutions.

## Table of Contents

- [Deployment Considerations](#deployment-considerations)
- [Model Export](#model-export)
- [Containerized Deployment](#containerized-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Edge Deployment](#edge-deployment)
- [API Development](#api-development)
- [Performance Optimization](#performance-optimization)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Security Considerations](#security-considerations)

## Deployment Considerations

Before deploying your SensorAugmentor model, consider the following:

### Hardware Requirements

- **CPU-only deployment**: Minimum 2 CPU cores, 4GB RAM
- **GPU acceleration**: NVIDIA GPU with CUDA support for larger models
- **Edge deployment**: Models can be quantized to run on devices with limited resources

### Software Requirements

- Python 3.8+
- PyTorch 1.9+ (or ONNX runtime for exported models)
- Docker (for containerized deployment)
- FastAPI/Flask (for API development)

### Data Flow

Consider how data will flow through your system:

1. **Data Collection**: How sensor data is collected and fed to the model
2. **Preprocessing**: How data is normalized and prepared for the model
3. **Inference**: How the model generates predictions
4. **Postprocessing**: How predictions are processed and delivered
5. **Storage**: How data and predictions are stored for future use

## Model Export

SensorAugmentor models can be exported in various formats for deployment:

### PyTorch Serialization

```python
import torch
from sensor_actuator_network import SensorAugmentor, ModelSerializer

# Create and train model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
# ... training code ...

# Save model with metadata using ModelSerializer
ModelSerializer.save_model(
    model=model,
    path="models/sensor_model.pt",
    metadata={
        "version": "1.0.0",
        "sensor_dim": 32,
        "hidden_dim": 64,
        "output_dim": 1,
        "normalization": {
            "mean_lq": dataset.mean_lq.tolist(),
            "std_lq": dataset.std_lq.tolist(),
            "mean_hq": dataset.mean_hq.tolist(),
            "std_hq": dataset.std_hq.tolist()
        }
    }
)

# Alternative: Standard PyTorch save
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {
        "sensor_dim": 32,
        "hidden_dim": 64,
        "output_dim": 1
    },
    "normalization": {
        "mean_lq": dataset.mean_lq,
        "std_lq": dataset.std_lq
    }
}, "models/sensor_model_standard.pt")
```

### ONNX Export

[ONNX](https://onnx.ai/) (Open Neural Network Exchange) allows deployment across different frameworks and platforms:

```python
import torch
from sensor_actuator_network import SensorAugmentor

# Create and load model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("models/sensor_model.pt")["model_state_dict"])
model.eval()

# Create dummy input for tracing
dummy_input = torch.randn(1, 32)

# Export to ONNX
torch.onnx.export(
    model,                                # model being run
    dummy_input,                          # model input
    "models/sensor_model.onnx",           # where to save the model
    export_params=True,                   # store the trained parameter weights inside the model file
    opset_version=11,                     # the ONNX version to export the model to
    do_constant_folding=True,             # optimization: fold constant values
    input_names=["sensor_input"],         # model input names
    output_names=["reconstructed_hq", "actuator_command"],  # model output names
    dynamic_axes={                         # variable length axes
        "sensor_input": {0: "batch_size"},
        "reconstructed_hq": {0: "batch_size"},
        "actuator_command": {0: "batch_size"}
    }
)

print("Model exported to ONNX format")
```

### TorchScript

[TorchScript](https://pytorch.org/docs/stable/jit.html) allows deployment in C++ environments:

```python
import torch
from sensor_actuator_network import SensorAugmentor

# Create and load model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("models/sensor_model.pt")["model_state_dict"])
model.eval()

# Convert to TorchScript via tracing
example_input = torch.randn(1, 32)
traced_model = torch.jit.trace(model, example_input)

# Save the traced model
traced_model.save("models/sensor_model_traced.pt")

# Alternatively, use scripting
scripted_model = torch.jit.script(model)
scripted_model.save("models/sensor_model_scripted.pt")

print("Model exported to TorchScript format")
```

## Containerized Deployment

Using Docker containers ensures your model runs consistently across different environments.

### Dockerfile

Create a `Dockerfile` in your project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/sensor_model.pt /app/models/
COPY sensor_actuator_network.py /app/
COPY api_server.py /app/

# Expose the port
EXPOSE 8000

# Start the API server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

For more complex setups, use Docker Compose:

```yaml
# docker-compose.yml
version: '3'

services:
  sensor_api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0  # For GPU use
      - MODEL_PATH=/app/models/sensor_model.pt
      - LOG_LEVEL=INFO
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### Building and Running the Container

```bash
# Build the image
docker build -t sensor-augmentor:v1.0 .

# Run the container
docker run -p 8000:8000 -v $(pwd)/models:/app/models sensor-augmentor:v1.0

# Or with docker-compose
docker-compose up -d
```

## Cloud Deployment

SensorAugmentor models can be deployed to various cloud providers.

### AWS SageMaker

```python
import sagemaker
from sagemaker.pytorch import PyTorchModel

# Set up SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create PyTorch model
pytorch_model = PyTorchModel(
    model_data='s3://my-bucket/sensor_model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='1.9.0',
    py_version='py38',
    predictor_cls=sagemaker.pytorch.PyTorchPredictor
)

# Deploy model to endpoint
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.xlarge'
)

# Example inference
response = predictor.predict({"sensor_data": [...]})
```

The `inference.py` script should implement the required SageMaker handlers:

```python
import torch
from sensor_actuator_network import SensorAugmentor
import json
import numpy as np

def model_fn(model_dir):
    """Load the model and normalization parameters."""
    checkpoint = torch.load(f"{model_dir}/model.pt")
    
    model = SensorAugmentor(
        sensor_dim=checkpoint["model_config"]["sensor_dim"],
        hidden_dim=checkpoint["model_config"]["hidden_dim"],
        output_dim=checkpoint["model_config"]["output_dim"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return {
        "model": model,
        "mean_lq": checkpoint["normalization"]["mean_lq"],
        "std_lq": checkpoint["normalization"]["std_lq"]
    }

def input_fn(request_body, request_content_type):
    """Parse input data."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        sensor_data = torch.tensor(data["sensor_data"], dtype=torch.float32)
        return sensor_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make prediction."""
    model = model_dict["model"]
    mean_lq = model_dict["mean_lq"]
    std_lq = model_dict["std_lq"]
    
    # Normalize input
    input_normalized = (input_data - mean_lq) / std_lq
    
    # Perform inference
    with torch.no_grad():
        reconstructed_hq, actuator_command, _, _ = model(input_normalized)
    
    return {
        "reconstructed_hq": reconstructed_hq.numpy().tolist(),
        "actuator_command": actuator_command.numpy().tolist()
    }

def output_fn(prediction, response_content_type):
    """Format output data."""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")
```

### Google Cloud AI Platform

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import Model

# Create model in Vertex AI
model = Model.upload(
    display_name="sensor-augmentor",
    artifact_uri="gs://my-bucket/sensor_model/",
    serving_container_image_uri="gcr.io/my-project/sensor-augmentor:v1"
)

# Deploy model to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3
)

# Example inference
response = endpoint.predict(
    instances=[{"sensor_data": [...]}]
)
```

### Azure Machine Learning

```python
from azureml.core import Workspace, Model, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Connect to workspace
ws = Workspace.from_config()

# Register model
model = Model.register(
    workspace=ws,
    model_path="models/sensor_model.pt",
    model_name="sensor-augmentor",
    description="SensorAugmentor model for enhancing sensor data"
)

# Set up environment
env = Environment.from_conda_specification(
    name="sensor-env",
    file_path="environment.yml"
)

# Create inference config
inference_config = InferenceConfig(
    entry_script="score.py",
    environment=env
)

# Deploy model
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    auth_enabled=True
)

service = Model.deploy(
    workspace=ws,
    name="sensor-augmentor-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config
)

service.wait_for_deployment(show_output=True)
```

## Edge Deployment

For resource-constrained devices, optimization is essential.

### Model Quantization

```python
import torch
from sensor_actuator_network import SensorAugmentor

# Load model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("models/sensor_model.pt")["model_state_dict"])
model.eval()

# Quantize the model
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Specify which layers to quantize
    dtype=torch.qint8   # Quantization type
)

# Save quantized model
torch.save(quantized_model.state_dict(), "models/sensor_model_quantized.pt")

# Verify model size reduction
original_size = os.path.getsize("models/sensor_model.pt") / 1024
quantized_size = os.path.getsize("models/sensor_model_quantized.pt") / 1024
print(f"Original model size: {original_size:.2f} KB")
print(f"Quantized model size: {quantized_size:.2f} KB")
print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%")
```

### Raspberry Pi Deployment

For deploying on Raspberry Pi or similar devices:

1. Install dependencies on the device:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev

# Install PyTorch (version compatible with your device)
pip3 install torch torchvision torchaudio

# Install other dependencies
pip3 install numpy fastapi uvicorn
```

2. Create a simple inference script (`inference.py`):
```python
import torch
from sensor_actuator_network import SensorAugmentor
import json
import numpy as np

# Load the model
model_path = "models/sensor_model_quantized.pt"
model_config = torch.load(model_path, map_location=torch.device('cpu'))
model = SensorAugmentor(
    sensor_dim=model_config["model_config"]["sensor_dim"],
    hidden_dim=model_config["model_config"]["hidden_dim"],
    output_dim=model_config["model_config"]["output_dim"]
)
model.load_state_dict(model_config["model_state_dict"])
model.eval()

# Load normalization parameters
mean_lq = model_config["normalization"]["mean_lq"]
std_lq = model_config["normalization"]["std_lq"]

def predict(sensor_data):
    """Process sensor data and return predictions."""
    # Convert to tensor
    sensor_tensor = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)
    
    # Normalize
    sensor_normalized = (sensor_tensor - mean_lq) / std_lq
    
    # Inference
    with torch.no_grad():
        reconstructed_hq, actuator_command, _, _ = model(sensor_normalized)
    
    return {
        "reconstructed_hq": reconstructed_hq.squeeze(0).numpy().tolist(),
        "actuator_command": actuator_command.squeeze(0).numpy().tolist()
    }

# Example usage
if __name__ == "__main__":
    # Simulated sensor reading
    sensor_reading = np.random.randn(32).tolist()
    result = predict(sensor_reading)
    print(json.dumps(result, indent=2))
```

3. Run the script:
```bash
python3 inference.py
```

### Arduino and Microcontroller Deployment

For extremely resource-constrained devices:

1. Convert model to TFLite or similar format
2. Use a library like TensorFlow Lite for Microcontrollers or TinyML

Example conversion to TFLite:
```python
import torch
import onnx
import tf
import tensorflow as tf

# First export to ONNX
torch.onnx.export(
    model,                                # model being run
    dummy_input,                          # model input
    "models/sensor_model.onnx",           # where to save the model
    export_params=True,
    opset_version=11,
    do_constant_folding=True
)

# Convert ONNX model to TensorFlow
import onnx_tf
tf_rep = onnx_tf.backend.prepare(onnx.load("models/sensor_model.onnx"))
tf_rep.export_graph("models/sensor_model_tf")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("models/sensor_model_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save TFLite model
with open("models/sensor_model.tflite", "wb") as f:
    f.write(tflite_model)
```

## API Development

Create a REST API for serving model predictions.

### FastAPI Implementation

Create a file named `api_server.py`:

```python
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sensor_actuator_network import SensorAugmentor, ModelSerializer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="SensorAugmentor API", 
              description="API for enhancing sensor data and generating actuator commands")

# Model path
MODEL_PATH = os.environ.get("MODEL_PATH", "models/sensor_model.pt")

# Load model
try:
    logger.info(f"Loading model from {MODEL_PATH}")
    model, metadata = ModelSerializer.load_model(MODEL_PATH)
    model.eval()
    
    # Get normalization parameters
    mean_lq = torch.tensor(metadata["normalization"]["mean_lq"])
    std_lq = torch.tensor(metadata["normalization"]["std_lq"])
    mean_hq = torch.tensor(metadata["normalization"]["mean_hq"])
    std_hq = torch.tensor(metadata["normalization"]["std_hq"])
    
    logger.info(f"Model loaded successfully. Version: {metadata.get('version', 'unknown')}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Model initialization failed: {str(e)}")

# Pydantic models for request/response
class SensorData(BaseModel):
    data: List[float]
    normalize: bool = True

class PredictionResponse(BaseModel):
    reconstructed_hq: List[float]
    actuator_command: List[float]
    latent_representation: List[float]
    meta: Dict[str, Any]

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "name": "SensorAugmentor API",
        "version": metadata.get("version", "unknown"),
        "model_info": {
            "sensor_dim": metadata.get("sensor_dim", "unknown"),
            "hidden_dim": metadata.get("hidden_dim", "unknown"),
            "output_dim": metadata.get("output_dim", "unknown")
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(sensor_data: SensorData):
    """Make a prediction based on sensor data."""
    try:
        # Convert to tensor
        data = torch.tensor(sensor_data.data, dtype=torch.float32).unsqueeze(0)
        
        # Check dimensions
        expected_dim = metadata.get("sensor_dim", len(sensor_data.data))
        if data.shape[1] != expected_dim:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected sensor data of dimension {expected_dim}, got {data.shape[1]}"
            )
        
        # Normalize if requested
        if sensor_data.normalize:
            data = (data - mean_lq) / std_lq
        
        # Inference
        with torch.no_grad():
            reconstructed_hq, actuator_command, latent_lq, _ = model(data)
        
        # Denormalize HQ reconstruction if needed
        if sensor_data.normalize:
            reconstructed_hq = reconstructed_hq * std_hq + mean_hq
        
        # Prepare response
        return PredictionResponse(
            reconstructed_hq=reconstructed_hq.squeeze(0).tolist(),
            actuator_command=actuator_command.squeeze(0).tolist(),
            latent_representation=latent_lq.squeeze(0).tolist(),
            meta={
                "normalized_input": sensor_data.normalize,
                "model_version": metadata.get("version", "unknown")
            }
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metadata")
def get_metadata():
    """Return model metadata."""
    return metadata

# Run with: uvicorn api_server:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Client Usage Example

```python
import requests
import numpy as np
import json

# Generate random sensor data
sensor_data = np.random.randn(32).tolist()

# Make API request
response = requests.post(
    "http://localhost:8000/predict",
    json={"data": sensor_data, "normalize": True}
)

# Process response
if response.status_code == 200:
    result = response.json()
    print("Reconstructed HQ signal:", result["reconstructed_hq"])
    print("Actuator command:", result["actuator_command"])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Performance Optimization

Optimize your deployment for better performance.

### Batch Processing

Processing data in batches improves throughput:

```python
import torch
from sensor_actuator_network import SensorAugmentor
import time

# Load model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("models/sensor_model.pt")["model_state_dict"])
model.eval()

# Prepare data: individual processing vs batch processing
num_samples = 1000
batch_size = 32
input_data = torch.randn(num_samples, 32)

# Measure individual processing time
start_time = time.time()
individual_results = []
with torch.no_grad():
    for i in range(num_samples):
        single_input = input_data[i:i+1]
        result = model(single_input)
        individual_results.append(result[0])  # Keep only reconstructed_hq
individual_time = time.time() - start_time

# Measure batch processing time
start_time = time.time()
batch_results = []
with torch.no_grad():
    for i in range(0, num_samples, batch_size):
        batch_input = input_data[i:i+batch_size]
        result = model(batch_input)
        batch_results.append(result[0])  # Keep only reconstructed_hq
batch_time = time.time() - start_time

# Compare
print(f"Individual processing time: {individual_time:.4f} seconds")
print(f"Batch processing time: {batch_time:.4f} seconds")
print(f"Speedup factor: {individual_time/batch_time:.2f}x")
```

### GPU Acceleration

Using GPUs can significantly improve performance:

```python
import torch
from sensor_actuator_network import SensorAugmentor
import time

# Load model
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
model.load_state_dict(torch.load("models/sensor_model.pt")["model_state_dict"])
model.eval()

# Prepare data
batch_size = 64
input_data = torch.randn(batch_size, 32)

# CPU inference
start_time = time.time()
with torch.no_grad():
    for _ in range(100):  # Run multiple times for more accurate measurement
        cpu_result = model(input_data)
cpu_time = (time.time() - start_time) / 100

# GPU inference (if available)
if torch.cuda.is_available():
    model.to("cuda")
    input_data_gpu = input_data.to("cuda")
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data_gpu)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            gpu_result = model(input_data_gpu)
    gpu_time = (time.time() - start_time) / 100
    
    print(f"CPU inference time: {cpu_time*1000:.2f} ms")
    print(f"GPU inference time: {gpu_time*1000:.2f} ms")
    print(f"Speedup factor: {cpu_time/gpu_time:.2f}x")
else:
    print("GPU not available for comparison")
```

## Monitoring and Maintenance

Monitor model performance in production.

### Basic Monitoring

```python
import time
import statistics
import torch
from sensor_actuator_network import SensorAugmentor

class ModelMonitor:
    def __init__(self, model, window_size=100):
        self.model = model
        self.window_size = window_size
        self.inference_times = []
        self.prediction_values = []
        self.ground_truth_values = []
        self.error_count = 0
        self.sample_count = 0
    
    def log_inference(self, input_data, ground_truth=None):
        """Log a single inference."""
        self.sample_count += 1
        
        try:
            # Time the inference
            start_time = time.time()
            with torch.no_grad():
                reconstructed_hq, actuator_command, _, _ = self.model(input_data)
            inference_time = time.time() - start_time
            
            # Store metrics
            self.inference_times.append(inference_time)
            if len(self.inference_times) > self.window_size:
                self.inference_times.pop(0)
            
            # Store prediction and ground truth if available
            if ground_truth is not None:
                error = torch.mean((reconstructed_hq - ground_truth)**2).item()
                self.prediction_values.append(reconstructed_hq.mean().item())
                self.ground_truth_values.append(ground_truth.mean().item())
                if len(self.prediction_values) > self.window_size:
                    self.prediction_values.pop(0)
                    self.ground_truth_values.pop(0)
            
            return {
                "inference_time": inference_time,
                "reconstructed_hq": reconstructed_hq,
                "actuator_command": actuator_command
            }
        except Exception as e:
            self.error_count += 1
            raise e
    
    def get_statistics(self):
        """Get current monitoring statistics."""
        stats = {
            "sample_count": self.sample_count,
            "error_rate": self.error_count / max(1, self.sample_count),
            "avg_inference_time": statistics.mean(self.inference_times) if self.inference_times else None,
            "p95_inference_time": sorted(self.inference_times)[int(len(self.inference_times)*0.95)] if len(self.inference_times) >= 20 else None,
        }
        
        if self.prediction_values and self.ground_truth_values:
            stats["prediction_mean"] = statistics.mean(self.prediction_values)
            stats["ground_truth_mean"] = statistics.mean(self.ground_truth_values)
        
        return stats

# Example usage
model = SensorAugmentor(sensor_dim=32, hidden_dim=64, output_dim=1)
monitor = ModelMonitor(model)

# Simulated monitoring
for i in range(1000):
    input_data = torch.randn(1, 32)
    result = monitor.log_inference(input_data)
    
    if i % 100 == 0:
        print(f"Statistics after {i+1} samples:")
        print(monitor.get_statistics())
```

### Integration with Monitoring Systems

For production, integrate with monitoring systems:

```python
import torch
import time
import logging
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Set up Prometheus metrics
INFERENCE_TIME = Histogram('inference_time_seconds', 'Time spent processing inference')
ERROR_COUNTER = Counter('inference_errors_total', 'Total number of inference errors')
PREDICTION_GAUGE = Gauge('prediction_value', 'Average prediction value')
REQUESTS_COUNTER = Counter('inference_requests_total', 'Total number of inference requests')

# Start Prometheus HTTP server
start_http_server(8001)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to process inference with monitoring
def process_inference(model, input_data):
    REQUESTS_COUNTER.inc()
    
    try:
        # Time the inference
        with INFERENCE_TIME.time():
            with torch.no_grad():
                reconstructed_hq, actuator_command, _, _ = model(input_data)
        
        # Track prediction value
        PREDICTION_GAUGE.set(reconstructed_hq.mean().item())
        
        return reconstructed_hq, actuator_command
    except Exception as e:
        ERROR_COUNTER.inc()
        logger.error(f"Inference error: {str(e)}")
        raise e
```

## Security Considerations

Protect your deployed models with these security measures:

### Input Validation

```python
def validate_sensor_data(data, expected_dim, min_value=-10, max_value=10):
    """Validate sensor data for security and data quality."""
    # Check dimensions
    if len(data) != expected_dim:
        raise ValueError(f"Expected {expected_dim} dimensions, got {len(data)}")
    
    # Check for NaN or Inf values
    if any(not np.isfinite(x) for x in data):
        raise ValueError("Data contains NaN or Infinite values")
    
    # Check range
    if any(x < min_value or x > max_value for x in data):
        raise ValueError(f"Data contains values outside range [{min_value}, {max_value}]")
    
    # Check for anomalies (e.g., using z-score)
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / max(std, 1e-6) for x in data]
    if any(abs(z) > 5 for z in z_scores):  # z-score > 5 is highly unusual
        logger.warning("Data contains outliers with z-score > 5")
    
    return True
```

### API Authentication

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# Authentication setup
SECRET_KEY = "your-secret-key"  # Store securely, e.g., in environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User model
class User(BaseModel):
    username: str
    disabled: bool = False

# Token model
class Token(BaseModel):
    access_token: str
    token_type: str

# In-memory user database (use a real DB in production)
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "hashed_password": pwd_context.hash("secret"),
        "disabled": False,
    }
}

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)
    return None

def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not pwd_context.verify(password, db[username]["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

# FastAPI example with authentication
app = FastAPI()

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict(sensor_data: SensorData, current_user: User = Depends(get_current_user)):
    # Your prediction logic here
    pass
```

---

This deployment guide covers the essentials for bringing your SensorAugmentor models to production. For more specific deployment scenarios or advanced configurations, please refer to the [examples directory](../examples/) or contact the maintainers. 