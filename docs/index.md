# SensorAugmentor Documentation

Welcome to the SensorAugmentor documentation. This framework is designed to enhance low-quality sensor data into high-quality signals and generate corresponding actuator control commands.

## Table of Contents

1. [Getting Started](getting_started.md)
2. [API Reference](api_reference.md)
3. [Examples](#examples)
4. [Testing](#testing)
5. [Contributing](#contributing)

## Overview

SensorAugmentor is a deep learning framework that bridges the gap between low-quality sensor data and high-precision control systems. By implementing a neural teacher-student architecture with residual connections, it:

1. **Enhances Signal Quality** - Transforms low-quality (LQ) sensor readings into high-quality (HQ) reconstructions
2. **Improves Latent Representations** - Creates refined embeddings optimized for downstream tasks
3. **Enables Precise Actuator Control** - Directly maps enhanced sensor data to optimal control commands

This framework has applications in various domains including:
- Industrial automation and robotics
- Autonomous vehicles
- Medical devices
- IoT and smart devices
- Environmental monitoring

## Examples

The SensorAugmentor framework includes several example implementations:

### Environmental Sensor Example

The [environmental sensor example](../examples/custom_dataset_example.py) demonstrates how to use the framework with environmental data (temperature, humidity, pressure, and light). It includes:

- Custom dataset implementation for environmental sensors
- Training workflow
- Visualization of sensor signals before and after enhancement

### Time-Series Sensor Example

The [time-series sensor example](../examples/time_series_example.py) shows how to work with sequential vibration data. It includes:

- Processing multiple time-series signals from different sensors
- Handling multi-dimensional sequential data
- Time-series visualization and evaluation

## Testing

The SensorAugmentor framework includes a comprehensive test suite to ensure code quality and reliability:

- **Unit Tests**: Test individual components (ResidualBlock, SensorAugmentor)
- **Integration Tests**: Test the complete training and inference workflow

To run the tests, use the `run_tests.py` script:

```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py unit

# Run only integration tests
python run_tests.py integration
```

## Contributing

Contributions to the SensorAugmentor framework are welcome! Here are some ways you can contribute:

- Implement new model architectures
- Add support for different types of sensor data
- Improve documentation and examples
- Fix bugs and issues
- Add new features

When contributing, please follow these guidelines:

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Submit a pull request with a clear description of your changes 