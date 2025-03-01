#!/usr/bin/env python
"""
Test runner for SensorAugmentor.

Usage:
    python run_tests.py                 # Run all tests
    python run_tests.py unit            # Run only unit tests
    python run_tests.py integration     # Run only integration tests
"""
import sys
import subprocess
import os


def run_tests(test_type=None):
    """Run the test suite with pytest."""
    base_cmd = [sys.executable, "-m", "pytest"]
    
    if test_type == "unit":
        base_cmd.extend(["tests/unit", "-v"])
        print("Running unit tests...")
    elif test_type == "integration":
        base_cmd.extend(["tests/integration", "-v"])
        print("Running integration tests...")
    else:
        base_cmd.extend(["-v"])
        print("Running all tests...")
    
    # Add coverage reporting if pytest-cov is installed
    try:
        import pytest_cov
        base_cmd.extend(["--cov=sensor_actuator_network", "--cov-report=term"])
        print("Coverage reporting enabled.")
    except ImportError:
        print("Coverage reporting disabled (pytest-cov not installed).")
    
    # Run the tests
    result = subprocess.run(base_cmd)
    return result.returncode


if __name__ == "__main__":
    # Set current directory to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Determine test type from command line argument
    test_type = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Run tests
    sys.exit(run_tests(test_type)) 