# Contributing to SensorAugmentor

Thank you for your interest in contributing to SensorAugmentor! This document provides guidelines and instructions for contributing to make the process smooth and effective for everyone involved.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) to understand what behaviors will and will not be tolerated.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Git
- Basic understanding of neural networks and sensor data processing

### Setup Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SensorAugmentor.git
   cd SensorAugmentor
   ```
3. Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/SensorAugmentor.git
   ```
4. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   pip install -r requirements-dev.txt
   ```

### First Contribution

Looking for a place to start?

1. Look for issues tagged with `good-first-issue` or `help-wanted`
2. Check the [roadmap](ROADMAP.md) for future features you might want to work on
3. Improve documentation or write tutorials
4. Add tests or improve test coverage

## Development Process

### Branches

- `main`: The main development branch. All features, fixes, and changes will be merged here.
- `release/X.Y.Z`: Release branches for specific versions
- For your contributions, create a feature branch from `main`:
  ```bash
  git checkout -b feature/your-feature-name
  ```

### Development Flow

1. Make sure your `main` branch is up to date:
   ```bash
   git checkout main
   git pull upstream main
   ```

2. Create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Make your changes and commit them with descriptive messages

4. Keep your branch updated with the main branch:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

5. Push your branch to your fork:
   ```bash
   git push -u origin feature/your-feature-name
   ```

6. Create a pull request

## Reporting Bugs

Before submitting a bug report:

1. Check the [issue tracker](https://github.com/ORIGINAL_OWNER/SensorAugmentor/issues) to see if it's already reported
2. Update your local repository to ensure the bug still exists in the latest version
3. Determine if it's a bug in SensorAugmentor or in a dependency

When submitting a bug report, include:

- Clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, PyTorch version, etc.)
- Screenshots or code snippets if applicable
- Any relevant logs or error messages

Use the bug report template when opening an issue.

## Feature Requests

Feature requests are welcome! When submitting feature requests:

1. Describe the feature in detail
2. Explain why this feature would be valuable to the project
3. Provide examples of how the feature would be used
4. If possible, outline a suggested implementation approach

Use the feature request template when opening an issue.

## Pull Request Process

1. Ensure your code follows our [coding standards](#coding-standards)
2. Include tests for new features or bug fixes
3. Update documentation to reflect any changes
4. Ensure all tests pass:
   ```bash
   python run_tests.py --all
   ```
5. Make sure your branch is updated with the latest changes from `main`
6. Create a pull request with a clear title and detailed description
7. Link any relevant issues in your pull request description using keywords like "Fixes #123" or "Resolves #456"

### Pull Request Review Process

- At least one maintainer will review your PR
- Feedback will be provided as comments on the PR
- Address any requested changes and push updates to the same branch
- Once approved, a maintainer will merge your PR

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these additional guidelines:

- Use 4 spaces for indentation (not tabs)
- Maximum line length is 88 characters (aligned with Black formatter)
- Use docstrings for all public classes, methods, and functions
- Use type hints where appropriate
- Format code with Black and isort

### Tool Configuration

We use the following tools for code quality:

- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For style guide enforcement
- **mypy**: For type checking

Configuration files for these tools are in the repository.

### Documentation Style

- Use Google-style docstrings
- Include type information in docstrings
- Document parameters, return values, and exceptions
- Include examples where appropriate

Example:
```python
def process_sensor_data(data: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Process raw sensor data for input to the model.
    
    Args:
        data: Raw sensor readings with shape (batch_size, sensor_dim)
        normalize: Whether to normalize the data
        
    Returns:
        Processed sensor data ready for model input
        
    Raises:
        ValueError: If data dimensions are incorrect
    
    Example:
        >>> raw_data = np.random.randn(10, 32)
        >>> processed = process_sensor_data(raw_data)
        >>> print(processed.shape)
        (10, 32)
    """
    # Implementation
```

## Testing

### Test Structure

- Unit tests: Test individual components in isolation
- Integration tests: Test multiple components working together
- Performance tests: Benchmark tests marked with the `benchmark` marker

### Running Tests

Run all tests:
```bash
python run_tests.py --all
```

Run specific test types:
```bash
python run_tests.py unit  # Run only unit tests
python run_tests.py integration  # Run only integration tests
python run_tests.py benchmark  # Run only benchmark tests
```

Run with coverage:
```bash
python run_tests.py --coverage
```

### Writing Tests

- Each test file should focus on a specific component or feature
- Use descriptive test names that explain what is being tested
- Use fixtures for common setup
- Use parameterized tests for testing multiple cases
- Mark slow tests with `@pytest.mark.slow`
- Include both positive and negative test cases

Example:
```python
import pytest
import torch
from sensor_actuator_network import ResidualBlock

class TestResidualBlock:
    @pytest.fixture
    def setup(self):
        return {"dim": 64, "batch_size": 32}
    
    def test_output_shape(self, setup):
        """Test that output shape matches input shape."""
        block = ResidualBlock(setup["dim"])
        x = torch.randn(setup["batch_size"], setup["dim"])
        out = block(x)
        assert out.shape == x.shape
```

## Documentation

### Documentation Structure

- **API Reference**: Document all public classes and functions
- **Tutorials**: Step-by-step guides for common tasks
- **How-to Guides**: Instructions for specific problems
- **Explanation**: Background information and concepts
- **Examples**: Working code examples

### Updating Documentation

When making changes that affect documentation:

1. Update docstrings in code
2. Update relevant Markdown files in the `docs/` directory
3. Add examples or tutorials for new features
4. Ensure links between documentation pages remain valid

## Community

- Join our [Discord server](https://discord.gg/sensoraugmentor) for real-time discussions
- Sign up for our [mailing list](https://groups.google.com/g/sensoraugmentor) for announcements
- Follow us on [Twitter](https://twitter.com/sensoraugmentor) for updates

### Recognition

All contributors will be recognized in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file and on the project website.

---

Thank you for contributing to SensorAugmentor! Your efforts help make this project better for everyone. 