#!/usr/bin/env python
"""
Test runner for SensorAugmentor.

Usage:
    python run_tests.py                 # Run all tests
    python run_tests.py unit            # Run only unit tests
    python run_tests.py integration     # Run only integration tests
    python run_tests.py benchmark       # Run only performance benchmark tests
    python run_tests.py --coverage      # Run all tests with coverage reporting
    python run_tests.py --slow          # Include slow tests that take longer to run
    python run_tests.py --all           # Run all tests including slow tests
"""
import sys
import subprocess
import os
import argparse


def run_tests(args):
    """Run the test suite with pytest."""
    base_cmd = [sys.executable, "-m", "pytest"]
    
    # Determine test type
    if args.test_type == "unit":
        base_cmd.extend(["tests/unit", "-v"])
        print("Running unit tests...")
    elif args.test_type == "integration":
        base_cmd.extend(["tests/integration", "-v"])
        print("Running integration tests...")
    elif args.test_type == "benchmark":
        base_cmd.extend(["-m", "benchmark", "-v"])
        print("Running performance benchmark tests...")
    else:
        base_cmd.extend(["-v"])
        print("Running all tests...")
    
    # Handle slow tests
    if not args.slow and args.test_type != "benchmark":
        base_cmd.extend(["-k", "not slow"])
        print("Skipping slow tests. Use --slow to include them.")
    
    # Add coverage reporting if requested
    if args.coverage:
        base_cmd.extend([
            "--cov=sensor_actuator_network",
            "--cov-report=term",
            "--cov-report=html:coverage_html"
        ])
        print("Coverage reporting enabled. HTML report will be in coverage_html/ directory.")
    
    # Run tests
    print(f"Running command: {' '.join(base_cmd)}")
    result = subprocess.run(base_cmd)
    
    # Print coverage summary if enabled
    if args.coverage and result.returncode == 0:
        print("\nCoverage Report:")
        try:
            with open("coverage_html/index.html", "r") as f:
                coverage_html = f.read()
                # Extract coverage percentage from HTML
                import re
                match = re.search(r'<span class="pc_cov">([\d]+)%</span>', coverage_html)
                if match:
                    coverage_percent = match.group(1)
                    print(f"Total coverage: {coverage_percent}%")
        except Exception as e:
            print(f"Could not read coverage report: {e}")
    
    return result.returncode


if __name__ == "__main__":
    # Set current directory to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SensorAugmentor tests")
    parser.add_argument("test_type", nargs="?", default=None, 
                        choices=["unit", "integration", "benchmark", None],
                        help="Type of tests to run (default: all)")
    parser.add_argument("--coverage", action="store_true", 
                        help="Enable coverage reporting")
    parser.add_argument("--slow", action="store_true", 
                        help="Include slow tests")
    parser.add_argument("--all", action="store_true", 
                        help="Run all tests including slow tests")
    
    args = parser.parse_args()
    
    # If --all is specified, include slow tests
    if args.all:
        args.slow = True
    
    # Run tests
    sys.exit(run_tests(args)) 