#!/bin/bash

# Installation script for RL Topology Optimization

echo "Installing dependencies for RL Topology Optimization..."

# Check if pip3 is available
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 not found. Please install Python 3 and pip3 first."
    exit 1
fi

# Install core dependencies
echo "Installing core dependencies..."
pip3 install torch>=2.0.0
pip3 install torch-geometric>=2.3.0
pip3 install stable-baselines3>=2.0.0
pip3 install gym>=0.26.0
pip3 install networkx>=3.1
pip3 install numpy>=1.24.0
pip3 install scipy>=1.10.0

# Install ATOP dependencies
echo "Installing ATOP dependencies..."
pip3 install sympy>=1.12
pip3 install pandas>=2.0.0

# Install optional dependencies
echo "Installing optional dependencies..."
pip3 install tensorboard>=2.13.0

# Install development dependencies
echo "Installing development dependencies..."
pip3 install pytest>=7.4.0

echo "Installation complete!"
echo ""
echo "To verify installation, run:"
echo "  cd rl_topo_opt && python3 test_basic.py"
